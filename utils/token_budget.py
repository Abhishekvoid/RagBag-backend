"""How long a piece of text is allowed to be before it reaches the embedder.

bge-small-en-v1.5 has a hard 512-token input window, and both backends enforce
it: TEI reports `max_input_length: 512`, and Cloudflare Workers AI rejects or
truncates past the same ceiling. Ingestion never approaches it — the chunker
caps every chunk at 200 tiktoken tokens, worst measured expansion 1.51x, so 303
bge tokens (see accounts/tests/test_embedding.py::ChunkTokenBudgetTests).

Chat queries are the unguarded path. They arrive as free text from a user and
are embedded verbatim, so the ceiling has to be enforced at the boundary.

Neither provider errors on an over-long input — both silently truncate to 512
(measured: a 782-token input returns HTTP 200, and Cloudflare reports
`usage.prompt_tokens: 512`). That is precisely why this check has to exist and
has to reject. Without it the failure is invisible: the user asks a long
question, the tail is discarded before it is ever embedded, and the assistant
answers the half of the question that survived, confidently and with citations.

WHY THIS IS NOT A CHARACTER LIMIT
---------------------------------
A tuned character limit is an observation about a sample, not a guarantee: the
same character count is a different number of tokens in English, in CJK, in a
URL, or in a run of punctuation. So the check is expressed in tokens, and it is
made in two independent ways — one for accuracy, one for proof.

GATE 1 — tiktoken, the tokenizer this project already ships
-----------------------------------------------------------
The exact bge tokenizer is a BERT WordPiece vocabulary, which lives in
`tokenizers`/`transformers`. This image deliberately ships neither: embedding
and reranking run as separate TEI services precisely so the web and worker
images carry no ML stack (see the header of requirements.txt). Asking TEI to
tokenize would mean a network round-trip inside request validation, against a
service the LEAN profile does not even run.

What the image does ship is tiktoken (cl100k_base), already used by the document
chunker. Measured bge/tiktoken expansion across six corpora:

    plain English prose          1.17x
    dense technical + code       1.51x   <- worst observed
    rare/scientific vocabulary   1.21x
    non-ASCII accented           1.04x
    CJK                          0.81x
    URLs and identifiers         1.14x

MAX_TIKTOKEN_TOKENS is set with SAFETY_RATIO = 2.0, a 32% margin over the worst
measured corpus, matching the ratio the chunker's tests already assert.

GATE 2 — the invariant that makes it a guarantee, not an estimate
------------------------------------------------------------------
Gate 1 is empirical, and an adversary is not a corpus. Runs of punctuation are
the cheap counter-example: cl100k merges `!!!!!!!!` into two or three tokens
while BERT splits punctuation one token per character. Measured against the real
bge tokenizer, a string of 480 exclamation marks is 60 tiktoken tokens and 482
bge tokens — a ratio of 8.03x, and 4x past what gate 1's 2.0 margin assumes.
Korean prose measures 1.79x, also above the 1.51x worst case sampling found. A
tiktoken-only limit is therefore not merely imprecise, it is unsound: on its own
it would have accepted roughly 2,000 bge tokens of punctuation.

WordPiece, however, has a structural floor. Every token it emits consumes at
least one non-whitespace character of the normalized input:

  * whitespace is dropped by the basic tokenizer and produces no token;
  * punctuation and CJK characters are split to at most one token each;
  * a word of n characters yields at most n subwords, since every single
    character is in the vocabulary ("##x" continuations consume input too);
  * a word longer than max_input_chars_per_word collapses to a single [UNK];
  * NFD normalization can decompose one character into several, so the count is
    taken *after* decomposition — an upper bound whether or not the model's
    tokenizer strips the resulting combining marks.

Therefore, for any text whatsoever:

    bge_tokens <= nfd_non_whitespace_chars(text) + 2      # [CLS] and [SEP]

That is a proof, not a measurement, and it holds for every language. Gate 2
enforces it directly, which is what lets us state that an accepted message
cannot exceed 512 bge tokens.

Both gates must pass. Gate 2 is the binding one for Latin scripts (English runs
~3.4 characters per bge token, so gate 2 is roughly 3x stricter than the model
truly requires); gate 1 binds for scripts tiktoken encodes densely. The price of
that conservatism is a shorter maximum question, and the way to buy it back is
to ship the real vocabulary — not to loosen either gate.

Verified across 18 corpora (English, CJK, Korean, Arabic, Hindi, accented Latin,
URLs, code, punctuation runs, long words, emoji, combining marks) by tokenizing
the largest accepted string of each with the real bge tokenizer via TEI: no
input passing both gates exceeded 482 tokens, against the limit of 512.

Nothing here truncates. Over-long input is rejected so the user can shorten it
themselves; silently cutting a question changes what was asked and returns a
confident answer to a question nobody posed.
"""

import unicodedata

# Hard limit of the model, shared by both providers.
BGE_MAX_TOKENS = 512

# [CLS] ... [SEP] are counted against that window.
BGE_SPECIAL_TOKENS = 2

# Tokens available to the text itself.
BGE_CONTENT_BUDGET = BGE_MAX_TOKENS - BGE_SPECIAL_TOKENS  # 510

# Gate 1. Worst measured bge/tiktoken expansion is 1.51x; 2.0 is the margin the
# chunker's tests already hold the ingestion path to.
SAFETY_RATIO = 2.0
MAX_TIKTOKEN_TOKENS = int(BGE_CONTENT_BUDGET / SAFETY_RATIO)  # 255

# Gate 2. The provable ceiling derived above.
MAX_NON_WHITESPACE_CHARS = BGE_CONTENT_BUDGET  # 510

_encoder = None


def _get_encoder():
    """cl100k_base, loaded once. Same encoding the document chunker uses."""
    global _encoder
    if _encoder is None:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tiktoken(text: str) -> int:
    return len(_get_encoder().encode(text or ""))


def count_non_whitespace(text: str) -> int:
    """Non-whitespace characters after NFD decomposition.

    Decomposing first is deliberate: it can only raise the count, and an upper
    bound that assumes the tokenizer keeps combining marks stays valid if the
    model is ever swapped for one that does not strip accents.
    """
    return sum(
        1 for ch in unicodedata.normalize("NFD", text or "") if not ch.isspace()
    )


def bge_token_upper_bound(text: str) -> int:
    """Largest number of tokens the real bge tokenizer could possibly produce."""
    return count_non_whitespace(text) + BGE_SPECIAL_TOKENS


def check_embedding_length(text: str):
    """Return None when `text` is safe to embed, else a human-readable reason.

    Returns a message rather than raising so the same rule can serve DRF field
    validation and the pipeline's internal defence without either importing the
    other's exception type.
    """
    if not text:
        return None

    chars = count_non_whitespace(text)
    if chars > MAX_NON_WHITESPACE_CHARS:
        return (
            f"Message is too long to search with: {chars} characters, "
            f"limit {MAX_NON_WHITESPACE_CHARS}. Please shorten it or ask about "
            "one thing at a time."
        )

    tokens = count_tiktoken(text)
    if tokens > MAX_TIKTOKEN_TOKENS:
        return (
            f"Message is too long to search with: about {tokens} tokens, "
            f"limit {MAX_TIKTOKEN_TOKENS}. Please shorten it or ask about one "
            "thing at a time."
        )

    return None


def is_safe_to_embed(text: str) -> bool:
    return check_embedding_length(text) is None


def _nfd_weight(ch: str) -> int:
    """How much a single character contributes to the bound in `count_non_whitespace`.

    Counted per character so a splitter can walk the original string while
    accounting in the same units the bound is expressed in. NFD decomposes each
    character independently, so summing per-character weights equals decomposing
    the whole string — a precomposed 'é' weighs 2, not 1, and the split stays
    conservative for accented text.
    """
    return sum(1 for c in unicodedata.normalize("NFD", ch) if not c.isspace())


def split_for_embedding(text: str, max_non_whitespace: int = MAX_NON_WHITESPACE_CHARS):
    """Cut `text` into pieces each provably within the model's 512-token window.

    Used by ingestion, where the input is a document rather than a question and
    discarding the overflow is not an option. The chunker counts cl100k tokens,
    which says nothing about WordPiece: a page of horizontal rules is 21 tiktoken
    tokens and 792 bge tokens (37x), and a chunk of ASCII table borders is 784.
    Those chunks are legal by every check the chunker applies and still exceed
    what the provider will accept.

    The split is LOSSLESS — ''.join(split_for_embedding(t)) == t for every input,
    including the whitespace at the seams. That property is what distinguishes
    this from the `chunk[:800]` truncation it replaces, which silently dropped
    the tail of any chunk that ran long, and it is asserted directly in the
    tests. Nothing is summarised, normalised, or thrown away; the same text comes
    back out, in order, in more pieces.

    Cuts prefer a whitespace boundary so ordinary prose splits between words, and
    fall back to a hard character cut when there is no whitespace to use — which
    is exactly the pathological case (rule lines, dot leaders) that makes this
    function necessary in the first place.
    """
    if not text:
        return []

    total = count_non_whitespace(text)
    if total <= max_non_whitespace:
        return [text]

    # Aim for equal pieces rather than filling each to the brim. Greedy filling
    # would cut a 700-character chunk into 510 + 190, leaving a runt fragment
    # with too little context to retrieve well; two pieces of ~350 carry the
    # same text and both remain searchable. Using the same piece count, this
    # only ever lowers the per-piece budget, so the guarantee is unaffected.
    piece_count = -(-total // max_non_whitespace)  # ceil
    budget = -(-total // piece_count)

    pieces = []
    start = 0
    n = len(text)

    while start < n:
        seen = 0
        last_ws = -1
        cut = n
        i = start

        while i < n:
            ch = text[i]
            if ch.isspace():
                last_ws = i
                i += 1
                continue

            weight = _nfd_weight(ch)
            # `i > start` guarantees forward progress: a single character wider
            # than the whole budget still gets a piece of its own rather than
            # looping forever.
            if seen + weight > budget and i > start:
                cut = i
                break

            seen += weight
            i += 1

        # Back off to the last space, but only when that does not throw away
        # more than half the window — otherwise a long unbroken run would split
        # into uselessly short pieces.
        if cut < n and last_ws > start and (last_ws - start) * 2 >= (cut - start):
            cut = last_ws + 1

        pieces.append(text[start:cut])
        start = cut

    return pieces
