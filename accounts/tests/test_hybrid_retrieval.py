"""Hybrid retrieval tests — fusion maths, degradation, and the router contract.

Nothing here touches Pinecone or an LLM. RRF is pure arithmetic over ranked
lists, which is exactly the kind of thing that should be pinned by tests rather
than eyeballed in a log, and the degradation paths are the ones that are
hardest to verify by hand because they only fire when something is broken.
"""

from types import SimpleNamespace
from unittest import mock

from django.test import TestCase

from accounts.rag_pipeline import (
    DEFAULT_INTENT,
    parse_contextualize_and_route,
)
from accounts.rag_service import (
    RRF_K,
    modality_balanced_weights,
    reciprocal_rank_fusion,
)
from utils.metrics.hybrid import HybridRetrievalStats
from utils.metrics.retrieval import mrr_at_k, recall_at_k


def hit(point_id, score=0.0):
    return SimpleNamespace(id=point_id, score=score, payload={"text": f"text-{point_id}"})


def ids(results):
    return [r.id for r in results]


class ReciprocalRankFusionTests(TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_single_list_preserves_its_order(self):
        ranked = [hit("a"), hit("b"), hit("c")]
        self.assertEqual(ids(reciprocal_rank_fusion([ranked])), ["a", "b", "c"])

    def test_score_is_the_rrf_sum_not_the_original(self):
        """The similarity score must be REPLACED. Leaving it in place invites a
        later sort-by-score to silently undo the fusion."""
        fused = reciprocal_rank_fusion([[hit("a", score=0.97)]])
        self.assertAlmostEqual(fused[0].score, 1.0 / (RRF_K + 1))

    def test_consensus_beats_a_single_first_place(self):
        """The property RRF exists for: agreement across lists outranks being
        top of one list. `b` is never first, but it is high in all three."""
        fused = reciprocal_rank_fusion([
            [hit("a"), hit("b")],
            [hit("c"), hit("b")],
            [hit("d"), hit("b")],
        ])
        self.assertEqual(fused[0].id, "b")

    def test_raw_scores_are_ignored_entirely(self):
        """A chunk with a huge cosine score from a sparse neighbourhood must
        not outrank a chunk that several lists agree on."""
        fused = reciprocal_rank_fusion([
            [hit("inflated", score=999.0)],
            [hit("agreed", score=0.1)],
            [hit("agreed", score=0.1)],
        ])
        self.assertEqual(fused[0].id, "agreed")

    def test_weights_shift_the_outcome(self):
        lists = [[hit("dense_pick")], [hit("sparse_pick")]]
        self.assertEqual(reciprocal_rank_fusion(lists, weights=[1.0, 5.0])[0].id,
                         "sparse_pick")
        self.assertEqual(reciprocal_rank_fusion(lists, weights=[5.0, 1.0])[0].id,
                         "dense_pick")

    def test_weight_count_must_match_list_count(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[hit("a")], [hit("b")]], weights=[1.0])

    def test_duplicate_within_one_list_votes_once(self):
        """A list must not be able to vote twice for the same document."""
        twice = reciprocal_rank_fusion([[hit("a"), hit("a")]])
        once = reciprocal_rank_fusion([[hit("a")]])
        self.assertEqual(len(twice), 1)
        self.assertAlmostEqual(twice[0].score, once[0].score)

    def test_items_without_an_id_are_skipped(self):
        fused = reciprocal_rank_fusion([[SimpleNamespace(id=None, score=1.0), hit("a")]])
        self.assertEqual(ids(fused), ["a"])

    def test_ties_break_deterministically(self):
        """Two runs over the same data must produce the same order, or an eval
        measures its own tie-shuffling."""
        lists = [[hit("z")], [hit("y")], [hit("x")]]
        first = ids(reciprocal_rank_fusion(lists))
        second = ids(reciprocal_rank_fusion(
            [[hit("x")], [hit("z")], [hit("y")]]
        ))
        self.assertEqual(first, second)
        self.assertEqual(first, ["x", "y", "z"])

    def test_payload_survives_fusion(self):
        fused = reciprocal_rank_fusion([[hit("a")]])
        self.assertEqual(fused[0].payload["text"], "text-a")


class ModalityWeightTests(TestCase):
    def test_each_modality_totals_one(self):
        dense_w, sparse_w = modality_balanced_weights(4, 1)
        self.assertAlmostEqual(sum(dense_w), 1.0)
        self.assertAlmostEqual(sum(sparse_w), 1.0)

    def test_expansion_count_does_not_shift_the_mix(self):
        """The bug this prevents: bumping expansions from 3 to 5 silently moving
        dense:sparse from 4:1 to 6:1, a change nobody decided on."""
        for dense_count in (1, 4, 6, 10):
            dense_w, sparse_w = modality_balanced_weights(dense_count, 1)
            self.assertAlmostEqual(sum(dense_w), sum(sparse_w))

    def test_sparse_can_outrank_dense_consensus_under_balanced_weights(self):
        """Unweighted, four dense lists outvote one sparse list 4:1 and a
        sparse-only hit could never surface. Balanced, it can."""
        dense_lists = [[hit("dense_pick")] for _ in range(4)]
        sparse_lists = [[hit("sparse_pick")]]
        dense_w, sparse_w = modality_balanced_weights(4, 1)

        balanced = reciprocal_rank_fusion(
            dense_lists + sparse_lists, weights=list(dense_w) + list(sparse_w)
        )
        self.assertEqual(balanced[0].score, balanced[1].score)

        unweighted = reciprocal_rank_fusion(dense_lists + sparse_lists)
        self.assertEqual(unweighted[0].id, "dense_pick")

    def test_missing_modality_yields_no_weights(self):
        dense_w, sparse_w = modality_balanced_weights(4, 0)
        self.assertEqual(len(dense_w), 4)
        self.assertEqual(sparse_w, [])


class EvalMetricTests(TestCase):
    def test_recall_counts_relevant_found(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a", "b"], k=3), 1.0)
        self.assertEqual(recall_at_k(["a", "x", "y"], ["a", "b"], k=3), 0.5)
        self.assertEqual(recall_at_k(["x", "y"], ["a"], k=2), 0.0)

    def test_recall_respects_the_cutoff(self):
        self.assertEqual(recall_at_k(["x", "a"], ["a"], k=1), 0.0)
        self.assertEqual(recall_at_k(["x", "a"], ["a"], k=2), 1.0)

    def test_mrr_is_reciprocal_of_first_hit(self):
        self.assertEqual(mrr_at_k(["a"], ["a"], k=8), 1.0)
        self.assertEqual(mrr_at_k(["x", "a"], ["a"], k=8), 0.5)
        self.assertAlmostEqual(mrr_at_k(["x", "y", "a"], ["a"], k=8), 1 / 3)
        self.assertEqual(mrr_at_k(["x", "y"], ["a"], k=8), 0.0)

    def test_mrr_discriminates_where_recall_saturates(self):
        """The regime this project is actually in: a chapter smaller than the
        funnel, so both arms return everything and only ORDER differs."""
        relevant = ["a"]
        worse, better = ["x", "y", "a"], ["a", "x", "y"]
        self.assertEqual(recall_at_k(worse, relevant, k=8),
                         recall_at_k(better, relevant, k=8))
        self.assertGreater(mrr_at_k(better, relevant, k=8),
                           mrr_at_k(worse, relevant, k=8))

    def test_no_relevant_ids_is_zero_not_a_crash(self):
        self.assertEqual(recall_at_k(["a"], [], k=8), 0.0)
        self.assertEqual(mrr_at_k(["a"], [], k=8), 0.0)


class HybridStatsTests(TestCase):
    def test_outcomes_are_counted_separately(self):
        stats = HybridRetrievalStats()
        stats.record(outcome="hybrid_ok")
        stats.record(outcome="sparse_empty")
        stats.record(outcome="sparse_unavailable")
        stats.record(outcome="disabled")

        summary = stats.get_summary()
        self.assertEqual(summary["queries"], 4)
        self.assertEqual(summary["hybrid_ok"], 1)
        self.assertEqual(summary["sparse_empty"], 1)
        self.assertEqual(summary["sparse_unavailable"], 1)
        self.assertEqual(summary["disabled"], 1)
        self.assertEqual(summary["dense_only_rate"], 0.75)

    def test_backfill_gap_and_outage_are_not_conflated(self):
        """Same symptom, opposite fixes — they must never share a counter."""
        stats = HybridRetrievalStats()
        stats.record(outcome="sparse_empty")
        summary = stats.get_summary()
        self.assertEqual(summary["sparse_empty"], 1)
        self.assertEqual(summary["sparse_unavailable"], 0)

    def test_unknown_outcome_shows_as_an_accounting_gap(self):
        stats = HybridRetrievalStats()
        stats.record(outcome="typo")
        summary = stats.get_summary()
        self.assertEqual(summary["queries"], 1)
        self.assertEqual(
            summary["hybrid_ok"] + summary["sparse_empty"]
            + summary["sparse_unavailable"] + summary["disabled"],
            0,
        )

    def test_empty_summary_does_not_divide_by_zero(self):
        self.assertEqual(HybridRetrievalStats().get_summary()["dense_only_rate"], 0.0)


class ContextualizeAndRouteParsingTests(TestCase):
    """Every malformed response must land on (original query, "question") —
    one degraded state, not one per failure mode."""

    def test_well_formed_response(self):
        raw = '{"standalone_question": "What is enthalpy?", "intent": "summary"}'
        question, intent = parse_contextualize_and_route(raw, "fallback")
        self.assertEqual(question, "What is enthalpy?")
        self.assertEqual(intent, "summary")

    def test_intent_outside_the_whitelist_defaults(self):
        raw = '{"standalone_question": "q", "intent": "chitchat"}'
        self.assertEqual(parse_contextualize_and_route(raw, "fallback")[1], DEFAULT_INTENT)

    def test_intent_is_case_insensitive(self):
        raw = '{"standalone_question": "q", "intent": "  SUMMARY  "}'
        self.assertEqual(parse_contextualize_and_route(raw, "fallback")[1], "summary")

    def test_invalid_json_falls_back_entirely(self):
        self.assertEqual(
            parse_contextualize_and_route("not json at all", "original"),
            ("original", DEFAULT_INTENT),
        )

    def test_empty_or_missing_question_falls_back_to_the_original(self):
        for raw in ('{"standalone_question": "", "intent": "question"}',
                    '{"intent": "question"}',
                    '{"standalone_question": null, "intent": "question"}'):
            self.assertEqual(parse_contextualize_and_route(raw, "original")[0], "original")

    def test_non_object_json_falls_back(self):
        for raw in ("[1, 2, 3]", '"a string"', "null"):
            self.assertEqual(
                parse_contextualize_and_route(raw, "original"),
                ("original", DEFAULT_INTENT),
            )

    def test_none_content_falls_back(self):
        """A reasoning model that spends its whole budget on hidden reasoning
        returns HTTP 200 with content=None."""
        self.assertEqual(
            parse_contextualize_and_route(None, "original"),
            ("original", DEFAULT_INTENT),
        )


class HybridSearchDegradationTests(TestCase):
    """The sparse half must never be able to fail a query."""

    def setUp(self):
        self.dense_lists = [[hit("a"), hit("b")]]

    def _run(self, **patches):
        from asgiref.sync import async_to_sync
        from accounts import rag_service

        defaults = {
            "HYBRID_SEARCH_ENABLED": True,
            "search_dense_ranked": mock.AsyncMock(return_value=self.dense_lists),
        }
        defaults.update(patches)

        with mock.patch.multiple(rag_service, **defaults):
            return async_to_sync(rag_service.hybrid_search)(
                [[0.1] * 384], query_text="q", filter=None
            )

    def test_sparse_outage_still_returns_dense_results(self):
        results = self._run(
            embed_sparse=mock.AsyncMock(side_effect=RuntimeError("pinecone down"))
        )
        self.assertEqual(ids(results), ["a", "b"])

    def test_sparse_query_failure_still_returns_dense_results(self):
        results = self._run(
            embed_sparse=mock.AsyncMock(return_value=[{"indices": [1], "values": [1.0]}]),
            search_sparse_ranked=mock.AsyncMock(side_effect=RuntimeError("timeout")),
        )
        self.assertEqual(ids(results), ["a", "b"])

    def test_empty_sparse_result_is_not_treated_as_an_outage(self):
        results = self._run(
            embed_sparse=mock.AsyncMock(return_value=[{"indices": [1], "values": [1.0]}]),
            search_sparse_ranked=mock.AsyncMock(return_value=[]),
        )
        self.assertEqual(ids(results), ["a", "b"])

    def test_disabled_skips_sparse_entirely(self):
        sparse = mock.AsyncMock()
        results = self._run(HYBRID_SEARCH_ENABLED=False, embed_sparse=sparse)
        sparse.assert_not_called()
        self.assertEqual(ids(results), ["a", "b"])

    def test_sparse_results_are_fused_in_when_available(self):
        results = self._run(
            embed_sparse=mock.AsyncMock(return_value=[{"indices": [1], "values": [1.0]}]),
            search_sparse_ranked=mock.AsyncMock(return_value=[hit("sparse_only")]),
        )
        self.assertIn("sparse_only", ids(results))

    def test_total_outage_preserves_the_dependency_error(self):
        with self.assertRaises(TimeoutError):
            self._run(
                search_dense_ranked=mock.AsyncMock(side_effect=TimeoutError("dense down")),
                embed_sparse=mock.AsyncMock(side_effect=RuntimeError("sparse down")),
            )

    def test_dense_outage_can_still_use_sparse_evidence(self):
        results = self._run(
            search_dense_ranked=mock.AsyncMock(side_effect=TimeoutError("dense down")),
            embed_sparse=mock.AsyncMock(return_value=[{"indices": [1], "values": [1.0]}]),
            search_sparse_ranked=mock.AsyncMock(return_value=[hit("sparse_only")]),
        )
        self.assertEqual(ids(results), ["sparse_only"])

    def test_all_dense_queries_failing_is_not_an_empty_search(self):
        from asgiref.sync import async_to_sync
        from accounts.rag_service import search_dense_ranked

        with mock.patch("accounts.rag_service._query_dense",
                        side_effect=TimeoutError("dense down")):
            with self.assertRaises(TimeoutError):
                async_to_sync(search_dense_ranked)([[0.1], [0.2]], filter=None)

    def test_partial_dense_failure_keeps_successful_queries(self):
        from asgiref.sync import async_to_sync
        from accounts.rag_service import search_dense_ranked

        with mock.patch("accounts.rag_service._query_dense",
                        side_effect=[TimeoutError("one query down"), {"matches": []}]):
            self.assertEqual(
                async_to_sync(search_dense_ranked)([[0.1], [0.2]], filter=None), [[]]
            )
