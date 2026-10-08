import json
from unittest import mock

from django.test import SimpleTestCase

from accounts.rag_pipeline import RagPipeline, validated_search_queries


class QueryExpansionTests(SimpleTestCase):
    def test_invalid_payloads_fall_back_to_exactly_the_original(self):
        for raw in [None, "broken{", '[]', '{}', '{"queries": "abc"}',
                    '{"queries": [123]}', '{"queries": []}',
                    json.dumps({"queries": ["valid"] * 15}),
                    json.dumps({"queries": ["xx"]}),
                    json.dumps({"queries": ["x" * 151]}),
                    '{"queries": ["   "]}']:
            with self.subTest(raw=raw):
                self.assertEqual(validated_search_queries(raw, "Original?", "Refined?"), ["Original?"])

    @mock.patch("accounts.rag_pipeline.is_safe_to_embed", return_value=True)
    def test_deduplication_includes_original_and_contextualized(self, _):
        raw = json.dumps({"queries": [" original? ", "REFINED?", "A new query"]})
        self.assertEqual(validated_search_queries(raw, "Original?", "Refined?"),
                         ["Original?", "Refined?", "A new query"])

    @mock.patch("accounts.rag_pipeline.is_safe_to_embed", return_value=True)
    def test_contextualized_query_shares_the_three_alternative_budget(self, _):
        queries = validated_search_queries(
            '{"queries": ["First query", "Second query", "Third query"]}',
            "Original?", "Refined?",
        )
        self.assertEqual(queries, ["Original?", "Refined?", "First query", "Second query"])

    def test_run_passes_submitted_question_through_contextualization(self):
        from asgiref.sync import async_to_sync
        from accounts.rag_pipeline import PipelineOutcome, PipelineResult

        pipeline = object.__new__(RagPipeline)
        pipeline.contextualize_and_route = mock.AsyncMock(return_value=("What is gravity?", "question"))
        pipeline.handle_rag_search = mock.AsyncMock(return_value=PipelineResult(PipelineOutcome.SUCCESS))
        with mock.patch("accounts.rag_pipeline.is_safe_to_embed", return_value=True):
            async_to_sync(pipeline.run)("How does it work?", [], "chapter", "user")
        self.assertEqual(pipeline.handle_rag_search.call_args.kwargs["original_query"], "How does it work?")
