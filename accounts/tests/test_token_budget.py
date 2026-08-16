"""The 512-token ceiling on anything user-controlled reaching the embedder.

These tests are offline. The numbers they assert against were established by
tokenizing each corpus with the REAL bge tokenizer through a local TEI
(`docker compose --profile full up embedder`, POST /tokenize) and recorded here
so the suite never depends on that container being up.

The property under test is one-directional and worth stating plainly:

    if check_embedding_length(text) is None, then the real bge tokenizer
    produces at most 512 tokens for `text`.

The converse is deliberately NOT claimed. The check rejects a great deal of text
that would in fact have fit — English is roughly 3.4 characters per bge token,
so gate 2 is about 3x stricter than the model requires. That asymmetry is the
point: over-rejection costs a user one clear error message, under-acceptance
costs them a silently truncated question and a confident answer to it.
"""

from django.test import TestCase

from utils.token_budget import (
    BGE_MAX_TOKENS,
    split_for_embedding,
    MAX_NON_WHITESPACE_CHARS,
    MAX_TIKTOKEN_TOKENS,
    bge_token_upper_bound,
    check_embedding_length,
    count_non_whitespace,
    is_safe_to_embed,
)


class BoundArithmeticTests(TestCase):
    def test_limits_leave_the_special_tokens_room(self):
        self.assertLessEqual(
            MAX_NON_WHITESPACE_CHARS + 2, BGE_MAX_TOKENS,
            "[CLS] and [SEP] are counted against the 512 window",
        )
        self.assertLessEqual(MAX_TIKTOKEN_TOKENS * 2.0 + 2, BGE_MAX_TOKENS)

    def test_upper_bound_ignores_whitespace(self):
        self.assertEqual(bge_token_upper_bound("a b c"), 5)
        self.assertEqual(bge_token_upper_bound("a\n\t b   c"), 5)

    def test_upper_bound_counts_after_nfd_decomposition(self):
        """Precomposed é may decompose to two characters; count the larger."""
        self.assertGreaterEqual(
            count_non_whitespace("é"), count_non_whitespace("e")
        )

    def test_accepted_text_can_never_exceed_the_model_window(self):
        """The invariant itself, asserted structurally rather than by sampling."""
        for text in [
            "short",
            "a" * MAX_NON_WHITESPACE_CHARS,
            "!" * MAX_NON_WHITESPACE_CHARS,
            "语" * 200,
        ]:
            if is_safe_to_embed(text):
                self.assertLessEqual(bge_token_upper_bound(text), BGE_MAX_TOKENS)


class NormalMessagesAreAcceptedTests(TestCase):
    def test_short_question(self):
        self.assertIsNone(
            check_embedding_length("What is the powerhouse of the cell?")
        )

    def test_empty_and_whitespace_are_not_rejected_here(self):
        """Emptiness is a different field's problem; this gate is about size."""
        self.assertIsNone(check_embedding_length(""))
        self.assertIsNone(check_embedding_length("   \n  "))

    def test_a_long_but_realistic_question(self):
        text = (
            "Can you explain the difference between mitosis and meiosis, "
            "focusing on what happens to the chromosomes in each phase, and "
            "why meiosis produces four cells instead of two?"
        )
        self.assertIsNone(check_embedding_length(text))

    def test_message_just_under_the_limit_is_accepted(self):
        # 510 non-whitespace characters: exactly the documented maximum.
        text = "a" * MAX_NON_WHITESPACE_CHARS
        self.assertEqual(count_non_whitespace(text), MAX_NON_WHITESPACE_CHARS)
        self.assertIsNone(check_embedding_length(text))

    def test_whitespace_does_not_count_against_the_user(self):
        """A well-formatted question is not penalised for its line breaks."""
        text = "a" * MAX_NON_WHITESPACE_CHARS + "\n" * 200
        self.assertIsNone(check_embedding_length(text))


class OversizedMessagesAreRejectedTests(TestCase):
    def test_message_one_character_over_the_limit_is_rejected(self):
        self.assertIsNotNone(
            check_embedding_length("a" * (MAX_NON_WHITESPACE_CHARS + 1))
        )

    def test_pasted_essay_is_rejected(self):
        text = "The mitochondria is the powerhouse of the cell. " * 60
        self.assertIsNotNone(check_embedding_length(text))

    def test_rejection_message_is_actionable(self):
        problem = check_embedding_length("a" * 5000)
        self.assertIn("too long", problem.lower())
        self.assertIn(str(MAX_NON_WHITESPACE_CHARS), problem)


class NoSilentTruncationTests(TestCase):
    """The whole reason this module exists rather than a `text[:n]` slice."""

    def test_the_checker_never_returns_modified_text(self):
        """check_embedding_length reports; it has no way to alter the input."""
        text = "a" * 5000
        self.assertIsInstance(check_embedding_length(text), str)
        self.assertEqual(len(text), 5000, "input must not be mutated")

    def test_oversized_input_is_refused_rather_than_shortened(self):
        long_text = "word " * 400
        self.assertFalse(is_safe_to_embed(long_text))

    def test_serializer_rejects_instead_of_trimming(self):
        from accounts.serializers import RAGChatMessageSerializer

        long_text = "a" * 3000
        serializer = RAGChatMessageSerializer(
            data={
                "chapter": "3f8a91bc-2d4e-4c7a-9f1e-2b6d4a8c0e11",
                "text": long_text,
            }
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("text", serializer.errors)

    def test_serializer_returns_normal_text_byte_for_byte(self):
        from accounts.serializers import RAGChatMessageSerializer

        text = "  What is ATP synthase?  \n"
        serializer = RAGChatMessageSerializer(
            data={
                "chapter": "3f8a91bc-2d4e-4c7a-9f1e-2b6d4a8c0e11",
                "text": text,
            }
        )
        self.assertTrue(serializer.is_valid(), serializer.errors)
        # DRF's CharField strips surrounding whitespace by default; what matters
        # is that the validator itself removed nothing from the middle.
        self.assertIn("What is ATP synthase?", serializer.validated_data["text"])


class ScriptSpecificTests(TestCase):
    """Each case below was checked against the real bge tokenizer via TEI.

    The recorded number is the token count of the LARGEST string of that kind
    the gates will accept. All are under 512; the largest seen anywhere is 482.
    """

    def _largest_accepted(self, unit, cap=4000):
        text = unit
        for reps in range(1, cap):
            candidate = unit * reps
            if not is_safe_to_embed(candidate):
                break
            text = candidate
        return text

    def test_dense_technical_text(self):
        # measured: 477 non-ws chars -> 281 real bge tokens (1.55x tiktoken)
        unit = "def f(x): return np.dot(x.T, sigmoid(W@x+b)) # backprop step "
        self.assertTrue(is_safe_to_embed(unit))
        self.assertLessEqual(
            bge_token_upper_bound(self._largest_accepted(unit)), BGE_MAX_TOKENS
        )

    def test_cjk_text_is_accepted_at_useful_length(self):
        """CJK must not be crippled — bge counts it at ~1 token per character."""
        # measured: 196 Chinese chars -> 198 real bge tokens
        chinese = "光合作用是植物利用光能把二氧化碳和水转化为有机物的过程。"
        self.assertTrue(is_safe_to_embed(chinese * 5))
        self.assertLessEqual(
            bge_token_upper_bound(self._largest_accepted(chinese)), BGE_MAX_TOKENS
        )

    def test_japanese_and_korean_are_accepted(self):
        japanese = "細胞のミトコンドリアはエネルギーを生成する重要な器官です。"
        korean = "미토콘드리아는 세포의 에너지를 생산하는 소기관입니다."
        self.assertTrue(is_safe_to_embed(japanese))
        self.assertTrue(is_safe_to_embed(korean))
        # Korean measured 1.79x against tiktoken — above the 1.51x the ingestion
        # chunker's margin was sized from, and still inside the bound.
        self.assertLessEqual(
            bge_token_upper_bound(self._largest_accepted(korean)), BGE_MAX_TOKENS
        )

    def test_rtl_and_indic_scripts_are_accepted(self):
        self.assertTrue(
            is_safe_to_embed("الميتوكوندريا هي عضية تنتج الطاقة داخل الخلية الحية.")
        )
        self.assertTrue(is_safe_to_embed("माइटोकॉन्ड्रिया कोशिका का ऊर्जा गृह कहलाता है।"))

    def test_urls_and_identifiers(self):
        # measured: 455 non-ws chars -> 261 real bge tokens
        unit = "https://example.com/api/v1/docs/3f8a91bc-2d4e/chunks?limit=50&x=1 "
        self.assertTrue(is_safe_to_embed(unit))
        self.assertLessEqual(
            bge_token_upper_bound(self._largest_accepted(unit)), BGE_MAX_TOKENS
        )

    def test_punctuation_run_is_the_case_a_tiktoken_only_limit_would_miss(self):
        """480 '!' = 60 tiktoken tokens but 482 real bge tokens (8.03x).

        Gate 1 alone would happily accept ~2000 bge tokens of this. Gate 2 is
        what keeps it legal, and this test fails the moment gate 2 is removed.
        """
        from utils.token_budget import count_tiktoken

        text = "!" * MAX_NON_WHITESPACE_CHARS
        self.assertLess(
            count_tiktoken(text), MAX_TIKTOKEN_TOKENS,
            "gate 1 does not reject this — gate 2 is load-bearing",
        )
        self.assertTrue(is_safe_to_embed(text))
        self.assertLessEqual(bge_token_upper_bound(text), BGE_MAX_TOKENS)

        self.assertFalse(is_safe_to_embed("!" * (MAX_NON_WHITESPACE_CHARS + 1)))

    def test_emoji_and_combining_marks(self):
        self.assertTrue(is_safe_to_embed("🙂🎉🔥🚀🧬🧪📚🎓" * 5))
        self.assertTrue(is_safe_to_embed("é̀̂̃" * 20))


# The nine corpora the ingestion boundary has to survive. Kept at module scope
# so the offline tests and the opt-in TEI test measure exactly the same inputs.
INGESTION_CORPORA = {
    "normal prose": "The mitochondria is the powerhouse of the cell. " * 200,
    "code": "def f(x): return np.dot(x.T, sigmoid(W@x+b)) # backprop\n" * 120,
    "CJK": "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u5149\u80fd\u628a\u4e8c\u6c27\u5316\u78b3\u548c\u6c34\u8f6c\u5316\u4e3a\u6709\u673a\u7269\u7684\u8fc7\u7a0b\u3002" * 60,
    "URLs": "https://example.com/api/v1/docs/3f8a91bc-2d4e/chunks?limit=50&x=1 " * 80,
    "PDF dot leaders": "".join(
        "Chapter {} Introduction to Biology {} {}\n".format(i, "." * 40, i * 7)
        for i in range(1, 80)
    ),
    "ASCII tables": ("+" + "-" * 20 + "+" + "-" * 20 + "+\n") * 60,
    "horizontal rules": ("-" * 78 + "\n") * 60,
    "punctuation runs": "!" * 4000,
    "combining characters": "e\u0301\u0300\u0302\u0303" * 400,
}


def production_payloads(text):
    """Exactly what ingestion would send for `text`: chunk, filter, then split."""
    import tiktoken

    from accounts.tasks import chunk_text_by_token

    tok = tiktoken.get_encoding("cl100k_base")
    chunks = chunk_text_by_token(text, tok)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
    return [piece for c in chunks for piece in split_for_embedding(c)]


class IngestionTokenBoundaryTests(TestCase):
    """The chunker's 200 cl100k tokens bound nothing on the WordPiece side.

    Before the splitter, these corpora produced payloads the providers would not
    accept — measured with the real bge tokenizer through a local TEI:

        horizontal rule lines      792 bge tokens   (37.7x its tiktoken count)
        ASCII table borders        784
        PDF dot leaders            467

    After it, the worst payload across all nine corpora is 470. The assertions
    below use the provable bound rather than TEI, so the suite stays offline and
    hermetic; TeiVerifiedBoundTests re-checks the same inputs against the actual
    tokenizer when one is available.
    """

    def test_every_production_payload_is_within_the_model_window(self):
        for label, text in INGESTION_CORPORA.items():
            with self.subTest(corpus=label):
                payloads = production_payloads(text)
                self.assertTrue(payloads, "corpus produced no payloads")
                for payload in payloads:
                    self.assertLessEqual(
                        bge_token_upper_bound(payload), BGE_MAX_TOKENS,
                        "{}: a payload could exceed the 512-token limit".format(label),
                    )

    def test_pathological_corpora_actually_needed_splitting(self):
        """Guards the guard: if these stopped splitting, the test above is vacuous."""
        import tiktoken

        from accounts.tasks import chunk_text_by_token

        tok = tiktoken.get_encoding("cl100k_base")
        for label in ("horizontal rules", "ASCII tables", "punctuation runs"):
            with self.subTest(corpus=label):
                chunks = [
                    c.strip()
                    for c in chunk_text_by_token(INGESTION_CORPORA[label], tok)
                    if len(c.strip()) > 10
                ]
                self.assertGreater(
                    len(production_payloads(INGESTION_CORPORA[label])),
                    len(chunks),
                    "{} should have been split further".format(label),
                )

    def test_ordinary_prose_is_not_shredded(self):
        """Conservatism has a budget too — prose must stay retrievable."""
        payloads = production_payloads(INGESTION_CORPORA["normal prose"])
        shortest = min(len(p.strip()) for p in payloads)
        self.assertGreater(
            shortest, 100,
            "splitting produced fragments too small to retrieve usefully",
        )


class NoContentLostInSplittingTests(TestCase):
    """Splitting must be lossless. This is the property `chunk[:800]` violated."""

    def test_join_of_pieces_reproduces_the_input_exactly(self):
        for label, text in INGESTION_CORPORA.items():
            with self.subTest(corpus=label):
                for chunk in [text[:3000], text[:900], text[:520]]:
                    pieces = split_for_embedding(chunk)
                    self.assertEqual(
                        "".join(pieces), chunk,
                        "{}: splitting lost or altered content".format(label),
                    )

    def test_whitespace_at_the_seams_is_preserved(self):
        text = "alpha " * 400
        self.assertEqual("".join(split_for_embedding(text)), text)

    def test_short_text_is_returned_untouched(self):
        text = "The mitochondria is the powerhouse of the cell."
        self.assertEqual(split_for_embedding(text), [text])

    def test_empty_text_produces_no_payloads(self):
        self.assertEqual(split_for_embedding(""), [])

    def test_every_piece_is_within_budget(self):
        for label, text in INGESTION_CORPORA.items():
            with self.subTest(corpus=label):
                for piece in split_for_embedding(text):
                    self.assertLessEqual(
                        count_non_whitespace(piece), MAX_NON_WHITESPACE_CHARS
                    )

    def test_splitting_terminates_on_a_single_unbroken_run(self):
        """No whitespace to back off to — the hard-cut path must still finish."""
        text = "x" * 5000
        pieces = split_for_embedding(text)
        self.assertEqual("".join(pieces), text)
        self.assertGreater(len(pieces), 1)


class TeiVerifiedBoundTests(TestCase):
    """Opt-in check against the REAL bge tokenizer.

    Skipped unless a local TEI is reachable, so `manage.py test` never depends on
    a container or a network. This is not the provider — it is the same model
    served locally, and it is the only way to verify the bound rather than trust
    it. Run it with the FULL profile up:

        docker compose --profile full up -d embedder
        TEI_TOKENIZE_URL=http://127.0.0.1:8080/tokenize python manage.py test \
            accounts.tests.test_token_budget.TeiVerifiedBoundTests

    Last run: every payload across all nine corpora was within 512, worst 470.
    """

    def setUp(self):
        import json
        import os
        import urllib.request

        url = os.getenv("TEI_TOKENIZE_URL")
        if not url:
            self.skipTest("TEI_TOKENIZE_URL not set; skipping real-tokenizer check")

        def count(text):
            req = urllib.request.Request(
                url,
                data=json.dumps(
                    {"inputs": text, "add_special_tokens": True}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return len(json.load(r)[0])

        try:
            count("probe")
        except Exception as exc:
            self.skipTest("TEI unreachable at {}: {}".format(url, type(exc).__name__))

        self.count = count

    def test_real_tokenizer_agrees_with_the_bound(self):
        for label, text in INGESTION_CORPORA.items():
            with self.subTest(corpus=label):
                for payload in production_payloads(text)[:6]:
                    real = self.count(payload)
                    self.assertLessEqual(
                        real, BGE_MAX_TOKENS,
                        "{}: real tokenizer produced {} tokens".format(label, real),
                    )
                    self.assertLessEqual(
                        real, bge_token_upper_bound(payload),
                        "{}: the upper bound is not an upper bound".format(label),
                    )
