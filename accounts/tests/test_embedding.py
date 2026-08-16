"""Embedding client tests — all HTTP is mocked, the real provider is never called.

The dimension check is the important one. A provider that silently returns
768-dim vectors would poison the Pinecone index irreversibly, and the failure
would surface much later as bad retrieval rather than as an error.
"""

from unittest import mock

import httpx
from django.test import TestCase
from tenacity import RetryError, wait_none

from utils import embedding as emb
from utils.embedding import (
    EmbeddingClient,
    EmbeddingDimensionMismatch,
    EmbeddingInputTooLong,
    EmbeddingPoolingMismatch,
    auth_headers,
    build_request,
    check_pooling,
    parse_response,
)
from utils.token_budget import bge_token_upper_bound

DIM = 384


def vec(dim=DIM, fill=0.1):
    return [fill] * dim


def async_run(coro):
    from asgiref.sync import async_to_sync

    return async_to_sync(lambda: coro)()


class RequestFormatTests(TestCase):
    def test_tei_request_shape(self):
        self.assertEqual(build_request(["a", "b"], "tei"), {"inputs": ["a", "b"]})

    def test_cloudflare_request_shape_pins_pooling(self):
        """Cloudflare defaults to MEAN; the indexed vectors are CLS.

        Measured against a local TEI serving the same weights, the same sentence
        scores cosine 0.999999 with pooling=cls and 0.932 with pooling=mean —
        below the 0.95 floor at which the existing index must not be reused.
        Both answers are 384 well-formed floats, so nothing else in this module
        can tell them apart. The parameter is the only thing that can.
        """
        self.assertEqual(
            build_request(["a", "b"], "cloudflare"),
            {"text": ["a", "b"], "pooling": "cls"},
        )

    def test_cloudflare_pooling_is_never_left_to_the_provider_default(self):
        self.assertIn("pooling", build_request(["a"], "cloudflare"))
        self.assertEqual(emb.EMBEDDING_POOLING, "cls")

    def test_openai_request_shape_includes_model(self):
        body = build_request(["a"], "openai")
        self.assertEqual(body["input"], ["a"])
        self.assertEqual(body["model"], "BAAI/bge-small-en-v1.5")

    def test_model_identifier_is_the_confirmed_one(self):
        self.assertEqual(emb.EMBEDDING_MODEL, "BAAI/bge-small-en-v1.5")


class ResponseParsingTests(TestCase):
    def test_parses_tei_bare_list(self):
        self.assertEqual(parse_response([vec(), vec()]), [vec(), vec()])

    def test_parses_cloudflare_envelope(self):
        payload = {"result": {"shape": [1, DIM], "data": [vec()]}, "success": True}
        self.assertEqual(parse_response(payload), [vec()])

    def test_parses_openai_envelope(self):
        payload = {"data": [{"embedding": vec(), "index": 0}]}
        self.assertEqual(parse_response(payload), [vec()])

    def test_cloudflare_failure_flag_raises(self):
        with self.assertRaises(ValueError):
            parse_response({"success": False, "errors": [{"message": "nope"}]})

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            parse_response({"unexpected": "shape"})


class PoolingGuardTests(TestCase):
    """Cloudflare echoes `result.pooling`; treat disagreement as a hard error.

    Verified live: the field is present on every response, and reflects what was
    asked for (and reports "mean" when the parameter is omitted).
    """

    def test_matching_pooling_passes(self):
        check_pooling(
            {"result": {"pooling": "cls", "data": [vec()]}}, "cloudflare"
        )

    def test_mismatched_pooling_raises(self):
        with self.assertRaises(EmbeddingPoolingMismatch):
            check_pooling(
                {"result": {"pooling": "mean", "data": [vec()]}}, "cloudflare"
            )

    def test_mean_pooled_response_is_rejected_by_the_client(self):
        """Wrong pooling must fail the call, not just fail a helper."""
        client = EmbeddingClient()
        client.cb = mock.MagicMock()
        client.cb.is_open.return_value = False
        request = httpx.Request("POST", "https://provider.test/embed")
        response = httpx.Response(
            200,
            json={"success": True, "result": {"pooling": "mean", "data": [vec()]}},
            request=request,
        )

        with mock.patch.object(emb, "EMBEDDING_PROVIDER", "cloudflare"), \
             mock.patch.object(
                 client.client, "post", new=mock.AsyncMock(return_value=response)
             ):
            with self.assertRaises(EmbeddingPoolingMismatch):
                async_run(client.embed_texts(["hello"]))

    def test_absent_pooling_field_is_tolerated(self):
        """TEI has no such field — pooling is fixed by the model config."""
        check_pooling([vec()], "tei")
        check_pooling({"result": {"data": [vec()]}}, "cloudflare")


class OversizedInputIsRefusedTests(TestCase):
    """No caller can reach the provider with input it would silently truncate.

    Both TEI and Cloudflare accept a 782-token input with HTTP 200 and quietly
    embed the first 512 tokens — verified live. Nothing downstream can tell that
    vector from a complete one, so the client refuses to make the request at all.
    Checking here rather than only at each call site is deliberate: it is the
    difference between an audit that has to be repeated whenever someone adds a
    caller, and one that cannot go stale.
    """

    def setUp(self):
        self.client = EmbeddingClient()
        self.client.cb = mock.MagicMock()
        self.client.cb.is_open.return_value = False

    def test_oversized_input_never_reaches_the_provider(self):
        post = mock.AsyncMock()
        with mock.patch.object(self.client.client, "post", new=post):
            with self.assertRaises(EmbeddingInputTooLong):
                async_run(self.client.embed_texts(["a" * 600]))
        post.assert_not_called()

    def test_punctuation_run_is_refused_despite_a_small_tiktoken_count(self):
        """1,000 '!' is ~125 cl100k tokens and ~1,002 bge tokens."""
        post = mock.AsyncMock()
        with mock.patch.object(self.client.client, "post", new=post):
            with self.assertRaises(EmbeddingInputTooLong):
                async_run(self.client.embed_texts(["!" * 1000]))
        post.assert_not_called()

    def test_one_bad_entry_fails_the_whole_batch(self):
        """A batch is all-or-nothing; a partially-sent batch is worse than none."""
        post = mock.AsyncMock()
        with mock.patch.object(self.client.client, "post", new=post):
            with self.assertRaises(EmbeddingInputTooLong):
                async_run(self.client.embed_texts(["fine", "b" * 600, "also fine"]))
        post.assert_not_called()

    def test_the_error_names_the_offending_position(self):
        with mock.patch.object(self.client.client, "post", new=mock.AsyncMock()):
            with self.assertRaises(EmbeddingInputTooLong) as ctx:
                async_run(self.client.embed_texts(["ok", "c" * 600]))
        self.assertIn("input 1", str(ctx.exception))

    def test_oversized_input_is_not_retried(self):
        """Retrying identical input would fail identically five times."""
        self.assertNotIn(EmbeddingInputTooLong, emb.EMBEDDING_ERRORS)

    def test_normal_batches_still_pass_through(self):
        post = mock.AsyncMock(
            return_value=httpx.Response(
                200,
                json=[vec(), vec()],
                request=httpx.Request("POST", "https://provider.test/embed"),
            )
        )
        with mock.patch.object(self.client.client, "post", new=post):
            out = async_run(
                self.client.embed_texts(["a normal chunk", "another one"])
            )
        self.assertEqual(len(out), 2)
        post.assert_called_once()

    def test_split_output_is_always_accepted_by_the_client(self):
        """The splitter and the guard must agree, or ingestion deadlocks."""
        from utils.token_budget import split_for_embedding

        for text in [
            "-" * 3000,
            "The mitochondria is the powerhouse of the cell. " * 100,
            "!" * 2000,
        ]:
            for piece in split_for_embedding(text):
                self.assertLessEqual(
                    bge_token_upper_bound(piece), emb.BGE_MAX_TOKENS
                )


class AuthHeaderTests(TestCase):
    def test_bearer_header_sent_when_key_present(self):
        with mock.patch.object(emb, "EMBEDDING_API_KEY", "test-key-value"):
            self.assertEqual(
                auth_headers(), {"Authorization": "Bearer test-key-value"}
            )

    def test_no_header_for_unauthenticated_self_hosted_tei(self):
        with mock.patch.object(emb, "EMBEDDING_API_KEY", ""):
            self.assertEqual(auth_headers(), {})


class EmbedTextsTests(TestCase):
    def setUp(self):
        self.client = EmbeddingClient()
        self.client.cb = mock.MagicMock()
        self.client.cb.is_open.return_value = False

    def _no_backoff(self):
        """Retries are real; their sleeps are not worth 80s of suite time."""
        return mock.patch.object(
            EmbeddingClient.embed_texts.retry, "wait", wait_none()
        )

    def _respond(self, json_body, status=200):
        request = httpx.Request("POST", "https://provider.test/embed")
        return httpx.Response(status, json=json_body, request=request)

    def test_successful_embedding_returns_384_dims(self):
        with mock.patch.object(
            self.client.client, "post", new=mock.AsyncMock(
                return_value=self._respond([vec()])
            )
        ):
            out = async_run(self.client.embed_texts(["hello"]))

        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), DIM)
        self.client.cb.record_success.assert_called_once()

    def test_auth_header_is_actually_sent(self):
        post = mock.AsyncMock(return_value=self._respond([vec()]))
        with mock.patch.object(emb, "EMBEDDING_API_KEY", "secret-key"), \
             mock.patch.object(self.client.client, "post", new=post):
            async_run(self.client.embed_texts(["hello"]))

        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer secret-key"
        )

    def test_wrong_dimension_fails_loudly(self):
        """768 dims must raise, never be truncated or padded to 384."""
        with mock.patch.object(
            self.client.client, "post", new=mock.AsyncMock(
                return_value=self._respond([vec(768)])
            )
        ):
            with self.assertRaises(EmbeddingDimensionMismatch) as ctx:
                async_run(self.client.embed_texts(["hello"]))

        self.assertIn("384", str(ctx.exception))
        self.assertIn("768", str(ctx.exception))

    def test_short_vector_also_fails(self):
        with mock.patch.object(
            self.client.client, "post", new=mock.AsyncMock(
                return_value=self._respond([vec(128)])
            )
        ):
            with self.assertRaises(EmbeddingDimensionMismatch):
                async_run(self.client.embed_texts(["hello"]))

    def _assert_retried_then_raised(self, post_mock, expected_cause):
        """tenacity has no reraise=True here, so callers see RetryError.

        Asserting the wrapper AND the underlying cause documents the actual
        contract rather than the one we might wish for.
        """
        with self._no_backoff(), mock.patch.object(
            self.client.client, "post", new=post_mock
        ):
            with self.assertRaises(RetryError) as ctx:
                async_run(self.client.embed_texts(["hello"]))

        cause = ctx.exception.last_attempt.exception()
        self.assertIsInstance(cause, expected_cause)
        self.assertEqual(post_mock.await_count, 5, "should exhaust 5 attempts")

    def test_http_401_retries_then_raises(self):
        self._assert_retried_then_raised(
            mock.AsyncMock(
                return_value=self._respond({"error": "unauthorized"}, status=401)
            ),
            httpx.HTTPStatusError,
        )

    def test_http_500_retries_then_raises(self):
        self._assert_retried_then_raised(
            mock.AsyncMock(
                return_value=self._respond({"error": "boom"}, status=500)
            ),
            httpx.HTTPStatusError,
        )

    def test_timeout_retries_then_raises(self):
        self._assert_retried_then_raised(
            mock.AsyncMock(side_effect=httpx.TimeoutException("timed out")),
            httpx.TimeoutException,
        )

    def test_open_circuit_short_circuits_without_http(self):
        self.client.cb.is_open.return_value = True
        post = mock.AsyncMock()
        with mock.patch.object(self.client.client, "post", new=post):
            with self.assertRaises(emb.EmbeddingServiceUnavailable):
                async_run(self.client.embed_texts(["hello"]))
        post.assert_not_called()

    def test_empty_input_makes_no_request(self):
        post = mock.AsyncMock()
        with mock.patch.object(self.client.client, "post", new=post):
            self.assertEqual(async_run(self.client.embed_texts([])), [])
        post.assert_not_called()

    def test_api_key_never_appears_in_logs(self):
        with mock.patch.object(emb, "EMBEDDING_API_KEY", "super-secret-value"), \
             mock.patch.object(
                 self.client.client, "post",
                 new=mock.AsyncMock(return_value=self._respond([vec()])),
             ):
            with self.assertLogs("utils.embedding", level="DEBUG") as logs:
                async_run(self.client.embed_texts(["hello"]))

        self.assertNotIn("super-secret-value", "\n".join(logs.output))


class ProductionConfigTests(TestCase):
    """A managed provider with no API key must fail at boot, not per request."""

    def _reload(self, env):
        import importlib
        import os

        import core.settings

        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            core.settings.sys, "argv", ["manage.py", "runserver"]
        ):
            return importlib.reload(core.settings)

    def _base_env(self):
        return {
            "DEBUG": "False",
            "SECRET_KEY": "test-only-not-a-real-secret",
            "REDIS_URL": "redis://redis.internal:6379/0",
            "DJANGO_ALLOWED_HOSTS": "api.example.com",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com",
            "PINECONE_API_KEY": "test-only",
        }

    def tearDown(self):
        import importlib

        import core.settings

        importlib.reload(core.settings)

    def test_missing_embedding_url_fails_loudly(self):
        from django.core.exceptions import ImproperlyConfigured

        env = self._base_env()
        env.update({"EMBEDDING_URL": "", "EMBEDDING_PROVIDER": "cloudflare"})
        with self.assertRaises(ImproperlyConfigured):
            self._reload(env)

    def test_managed_provider_without_api_key_fails_loudly(self):
        from django.core.exceptions import ImproperlyConfigured

        env = self._base_env()
        env.update({
            "EMBEDDING_URL": "https://provider.test/embed",
            "EMBEDDING_PROVIDER": "cloudflare",
            "EMBEDDING_API_KEY": "",
        })
        with self.assertRaises(ImproperlyConfigured):
            self._reload(env)

    def test_unknown_provider_rejected(self):
        from django.core.exceptions import ImproperlyConfigured

        env = self._base_env()
        env.update({
            "EMBEDDING_URL": "https://provider.test/embed",
            "EMBEDDING_PROVIDER": "definitely-not-a-provider",
        })
        with self.assertRaises(ImproperlyConfigured):
            self._reload(env)

    def test_valid_lean_config_boots(self):
        env = self._base_env()
        env.update({
            "EMBEDDING_URL": "https://provider.test/embed",
            "EMBEDDING_PROVIDER": "cloudflare",
            "EMBEDDING_API_KEY": "a-key",
            "RERANK_URL": "",
        })
        reloaded = self._reload(env)

        self.assertEqual(reloaded.EMBEDDING_PROVIDER, "cloudflare")
        self.assertFalse(reloaded.RERANK_ENABLED, "LEAN must ship with rerank off")

    def test_self_hosted_tei_needs_no_api_key(self):
        env = self._base_env()
        env.update({
            "EMBEDDING_URL": "http://embedder:80/embed",
            "EMBEDDING_PROVIDER": "tei",
            "EMBEDDING_API_KEY": "",
        })
        reloaded = self._reload(env)
        self.assertEqual(reloaded.EMBEDDING_PROVIDER, "tei")


class PineconeCompatibilityTests(TestCase):
    def test_index_dimension_still_384(self):
        from accounts import ai_clients

        self.assertEqual(
            ai_clients.EMBEDDING_DIM, DIM,
            "changing this requires a NEW Pinecone index and a full reindex",
        )

    def test_client_and_index_agree_on_dimension(self):
        from accounts import ai_clients

        self.assertEqual(ai_clients.EMBEDDING_DIM, emb.EXPECTED_DIM)


class RerankerDisabledTests(TestCase):
    """LEAN behaviour: RERANK_URL unset -> fallback, no HTTP, no exception."""

    def test_unset_rerank_url_returns_none_without_http(self):
        from utils import tei_rerank

        post = mock.AsyncMock()
        with mock.patch.object(tei_rerank, "RERANK_URL", ""), \
             mock.patch.object(tei_rerank.rerank_client, "_post", new=post):
            result = async_run(
                tei_rerank.rerank_client.rerank("q", ["a", "b"])
            )

        self.assertIsNone(result, "disabled reranker must signal fallback")
        post.assert_not_called()

    def test_empty_texts_returns_empty_list(self):
        from utils import tei_rerank

        self.assertEqual(
            async_run(tei_rerank.rerank_client.rerank("q", [])), []
        )

    def test_enabled_but_failing_reranker_still_falls_back(self):
        from utils import tei_rerank

        with mock.patch.object(tei_rerank, "RERANK_URL", "http://reranker:80/rerank"), \
             mock.patch.object(
                 tei_rerank.rerank_client, "_post",
                 new=mock.AsyncMock(side_effect=httpx.ConnectError("down")),
             ):
            result = async_run(tei_rerank.rerank_client.rerank("q", ["a"]))

        self.assertIsNone(result)


class ChunkTokenBudgetTests(TestCase):
    """Guard the 512-token ceiling imposed by the embedding model.

    The chunker counts cl100k_base (tiktoken) tokens; bge-small-en-v1.5 counts
    BERT WordPiece tokens. Different vocabularies, different counts — so the
    chunker's "200 tokens" is NOT 200 tokens to the model.

    Measured against the real bge tokenizer (TEI /tokenize) across six corpora:

        plain English prose          1.17x   ->  234 bge tokens
        dense technical + code       1.51x   ->  303 bge tokens   <- worst
        rare/scientific vocabulary   1.21x   ->  242
        non-ASCII accented           1.04x   ->  208
        CJK                          0.81x   ->  163
        URLs and identifiers         1.14x   ->  228

    Worst observed 303 on natural-language corpora, leaving 209 tokens of
    headroom. That margin holds for prose, and NOT in general — measuring the
    real bge tokenizer against degenerate but entirely realistic document text
    shows the ratio is not bounded by anything sampling can establish:

        PDF table-of-contents dot leaders   4.06x  ->  467 bge tokens
        ASCII table borders                 8.52x  ->  784
        horizontal rule lines              37.71x  ->  792   <- worst
        Korean prose                        1.70x  ->  340

    The last two exceed 512. The consequence is bounded rather than fatal:
    tasks.py already caps each chunk at MAX_CHUNK_LEN=800 characters, and
    WordPiece cannot emit more tokens than there are non-whitespace characters,
    so no chunk can exceed ~802 tokens; and both providers truncate over-long
    input to exactly 512 rather than erroring, identically, so TEI and
    Cloudflare stay in the same vector space and the index is never corrupted.
    What is lost is the tail of a chunk made almost entirely of punctuation,
    which carries no retrievable meaning in the first place.

    Chat queries get a real guarantee instead of this reasoning, because they
    are user-controlled and a truncated question is a wrong question — see
    utils/token_budget.py.

    These tests use tiktoken only (no network, no TEI) and fail if someone
    raises chunk_size past the point where the prose margin holds.
    """

    # The ratio beyond which the budget breaks, given chunk_size.
    BGE_LIMIT = 512
    OBSERVED_WORST_RATIO = 1.51
    SAFETY_RATIO = 2.0  # margin over the worst corpus we could construct

    def _tokenizer(self):
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")

    def test_default_chunk_size_is_within_budget(self):
        import inspect

        from accounts.tasks import chunk_text_by_token

        params = inspect.signature(chunk_text_by_token).parameters
        chunk_size = params["chunk_size"].default

        self.assertLessEqual(
            chunk_size * self.SAFETY_RATIO,
            self.BGE_LIMIT,
            f"chunk_size={chunk_size} leaves no safe margin under the model's "
            f"{self.BGE_LIMIT}-token limit (worst measured expansion "
            f"{self.OBSERVED_WORST_RATIO}x). Re-measure before raising it.",
        )

    def test_every_chunk_respects_the_tiktoken_budget(self):
        from accounts.tasks import chunk_text_by_token

        tok = self._tokenizer()
        corpora = [
            "The mitochondria is the powerhouse of the cell. " * 200,
            "def f(x): return np.dot(x.T, sigmoid(x)) # backprop " * 200,
            "Pneumonoultramicroscopicsilicovolcanoconiosis immunohistochemistry " * 200,
            "https://example.com/api/v1/documents/3f8a91bc-2d4e/chunks?limit=50 " * 200,
        ]

        for text in corpora:
            for chunk in chunk_text_by_token(text, tok):
                n = len(tok.encode(chunk))
                self.assertLessEqual(n, 200, "chunker exceeded its own chunk_size")
                self.assertLessEqual(
                    n * self.SAFETY_RATIO,
                    self.BGE_LIMIT,
                    "chunk could exceed the model's 512-token limit",
                )

    def test_overlap_cannot_grow_a_chunk(self):
        """chunk_overlap shrinks the stride, it must never widen a window."""
        import inspect

        from accounts.tasks import chunk_text_by_token

        params = inspect.signature(chunk_text_by_token).parameters
        self.assertLess(
            params["chunk_overlap"].default,
            params["chunk_size"].default,
            "overlap >= chunk_size would loop forever and never advance",
        )
