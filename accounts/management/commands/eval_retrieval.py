"""Measure retrieval quality against a hand-labelled golden set.

Runs the SAME query through two arms and compares them:

    dense    the 4 query variants against the dense index, RRF-fused
    hybrid   the same, plus the original query against the sparse index

Both arms are RRF-fused, which is deliberate. The only difference between them
is the presence of the sparse ranking, so any change in the numbers is
attributable to the sparse half rather than to swapping score-sorting for RRF
at the same time. Measuring one variable at a time is the whole point of having
a harness.

REPORTED METRICS

    recall@k   did retrieval find the material at all
    MRR@k      did it rank the material highly

Both are needed. When a chapter holds fewer chunks than the funnel retrieves,
recall saturates at 1.0 for both arms and stops discriminating — every chunk is
returned regardless of strategy. MRR keeps measuring in that regime because it
reads position, not membership. A chapter large enough for recall to move is
the one that tells you whether hybrid retrieval helps; run `--stats` to see
whether you have one.

WHAT THIS DOES NOT MEASURE

Answer quality. It measures what reaches the model, not what the model does
with it. That is the correct scope: everything downstream is confounded by
prompt and sampling temperature.

GOLDEN SET FORMAT (JSON list)

    [
      {
        "query": "what is the activation energy of the reaction",
        "chapter_id": "…",
        "user_id": "…",
        "relevant": ["<document_id>#12", "<document_id>#13"]
      }
    ]

`relevant` holds point ids. Use `--stats` to list the indexed chunks for a
chapter with their ids and text previews, which is how you build the set
without guessing.
"""

import json
from pathlib import Path

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError

from accounts.rag_service import (
    embed_texts,
    reciprocal_rank_fusion,
    search_dense_ranked,
    hybrid_search,
)
from utils.metrics.retrieval import mrr_at_k, recall_at_k

DEFAULT_K = 8
LIMIT_PER_VECTOR = 15


class Command(BaseCommand):
    help = "Compare dense-only vs hybrid+RRF retrieval on a golden set."

    def add_arguments(self, parser):
        parser.add_argument("--golden-set", help="Path to the golden set JSON file.")
        parser.add_argument("--k", type=int, default=DEFAULT_K,
                            help=f"Cutoff for recall@k / MRR@k (default {DEFAULT_K}).")
        parser.add_argument("--stats", metavar="CHAPTER_ID",
                            help="List indexed chunks for a chapter, to build a golden set.")
        parser.add_argument("--user", metavar="USER_ID",
                            help="User id, required with --stats.")

    def handle(self, *args, **options):
        if options["stats"]:
            return self._stats(options["stats"], options["user"])

        if not options["golden_set"]:
            raise CommandError("Provide --golden-set PATH (or --stats CHAPTER_ID).")

        path = Path(options["golden_set"])
        if not path.exists():
            raise CommandError(f"No such file: {path}")

        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CommandError(f"Golden set is not valid JSON: {e}")

        if not isinstance(cases, list) or not cases:
            raise CommandError("Golden set must be a non-empty JSON list.")

        k = options["k"]
        results = {"dense": [], "hybrid": []}
        skipped = []

        for i, case in enumerate(cases, start=1):
            missing = [f for f in ("query", "chapter_id", "user_id", "relevant")
                       if not case.get(f)]
            if missing:
                skipped.append((i, f"missing {', '.join(missing)}"))
                continue

            query = case["query"]
            from accounts.ingestion_versions import active_document_filter
            search_filter = active_document_filter(case["user_id"], case["chapter_id"])
            if search_filter is None:
                skipped.append((i, "No active document versions"))
                continue
            relevant = case["relevant"]

            try:
                dense_ids, hybrid_ids = async_to_sync(self._run_both)(
                    query, search_filter
                )
            except Exception as e:
                skipped.append((i, f"{type(e).__name__}: {e}"))
                self.stdout.write(self.style.ERROR(f"  case {i} failed: {e}"))
                continue

            for arm, ids in (("dense", dense_ids), ("hybrid", hybrid_ids)):
                results[arm].append({
                    "recall": recall_at_k(ids, relevant, k),
                    "mrr": mrr_at_k(ids, relevant, k),
                })

            d, h = results["dense"][-1], results["hybrid"][-1]
            marker = "  " if h["mrr"] == d["mrr"] else ("↑ " if h["mrr"] > d["mrr"] else "↓ ")
            self.stdout.write(
                f"{marker}case {i:>3}  dense mrr={d['mrr']:.3f} recall={d['recall']:.3f}"
                f"   hybrid mrr={h['mrr']:.3f} recall={h['recall']:.3f}   {query[:48]}"
            )

        self._report(results, k, len(cases), skipped)

    async def _run_both(self, query: str, search_filter: dict):
        """Both arms, sharing one set of dense embeddings.

        The expansion step is skipped here on purpose. It is an LLM call with a
        temperature, so it returns different queries on different runs — the
        baseline and the treatment would not be measuring the same retrieval.
        A harness that reshuffles its own inputs cannot attribute a delta to
        the thing under test.
        """
        embeddings = await embed_texts([query])

        dense_lists = await search_dense_ranked(
            embeddings, search_filter, LIMIT_PER_VECTOR
        )
        weights = [1.0 / len(dense_lists)] * len(dense_lists) if dense_lists else []
        dense_fused = reciprocal_rank_fusion(dense_lists, weights=weights)

        hybrid_fused = await hybrid_search(
            embeddings,
            query_text=query,
            filter=search_filter,
            limit_per_vector=LIMIT_PER_VECTOR,
        )

        return (
            [r.id for r in dense_fused],
            [r.id for r in hybrid_fused],
        )

    def _report(self, results, k, total_cases, skipped):
        scored = len(results["dense"])
        self.stdout.write("")

        if not scored:
            self.stdout.write(self.style.ERROR("No cases scored."))
            self._report_skips(skipped)
            return

        def mean(arm, metric):
            return sum(r[metric] for r in results[arm]) / len(results[arm])

        d_recall, d_mrr = mean("dense", "recall"), mean("dense", "mrr")
        h_recall, h_mrr = mean("hybrid", "recall"), mean("hybrid", "mrr")

        self.stdout.write(f"{scored}/{total_cases} cases scored, k={k}")
        self.stdout.write("")
        self.stdout.write(f"{'arm':<10}{'recall@'+str(k):>12}{'MRR@'+str(k):>12}")
        self.stdout.write("-" * 34)
        self.stdout.write(f"{'dense':<10}{d_recall:>12.4f}{d_mrr:>12.4f}")
        self.stdout.write(f"{'hybrid':<10}{h_recall:>12.4f}{h_mrr:>12.4f}")
        self.stdout.write("-" * 34)
        self.stdout.write(
            f"{'delta':<10}{h_recall - d_recall:>+12.4f}{h_mrr - d_mrr:>+12.4f}"
        )
        self.stdout.write("")

        # State the limits of the measurement in the output, so the number
        # cannot be quoted later without them.
        if d_recall >= 0.999 and h_recall >= 0.999:
            self.stdout.write(self.style.WARNING(
                "Recall is saturated in BOTH arms: retrieval is returning every "
                "relevant chunk regardless of strategy, because these chapters "
                "hold fewer chunks than the funnel retrieves. Recall cannot "
                "discriminate here — read the MRR column, and index a larger "
                "document before quoting a recall figure."
            ))
        if scored < 20:
            self.stdout.write(self.style.WARNING(
                f"Only {scored} cases. Treat the delta as directional, not "
                f"significant; ~25+ is the minimum worth quoting."
            ))

        self._report_skips(skipped)

    def _report_skips(self, skipped):
        if skipped:
            self.stdout.write(self.style.WARNING(f"\n{len(skipped)} case(s) skipped:"))
            for i, reason in skipped:
                self.stdout.write(self.style.WARNING(f"  case {i}: {reason}"))

    def _stats(self, chapter_id, user_id):
        """Dump a chapter's indexed chunks so a golden set can be written
        against real ids instead of guessed ones."""
        if not user_id:
            raise CommandError("--stats also needs --user USER_ID")

        from accounts.ai_clients import EMBEDDING_DIM, get_pinecone_index

        index = get_pinecone_index()
        response = index.query(
            vector=[0.0] * (EMBEDDING_DIM - 1) + [1.0],
            top_k=1000,
            include_metadata=True,
            include_values=False,
            filter={
                "user_id": {"$eq": str(user_id)},
                "chapter_id": {"$eq": str(chapter_id)},
            },
        )
        matches = (
            response.get("matches") if isinstance(response, dict)
            else getattr(response, "matches", [])
        ) or []

        if not matches:
            self.stdout.write(self.style.WARNING(
                "No chunks indexed for that chapter/user. If the document was "
                "uploaded before the hybrid migration, run `manage.py reindex_hybrid`."
            ))
            return

        # Sort by chunk index so the output reads in document order rather than
        # in whatever order an arbitrary probe vector happened to rank them.
        def chunk_index(match):
            raw_id = match["id"] if isinstance(match, dict) else match.id
            tail = str(raw_id).rsplit("#", 1)[-1]
            return int(tail) if tail.isdigit() else 1 << 30

        self.stdout.write(f"{len(matches)} chunk(s) indexed:\n")
        for match in sorted(matches, key=chunk_index):
            raw_id = match["id"] if isinstance(match, dict) else match.id
            meta = (match.get("metadata") if isinstance(match, dict)
                    else getattr(match, "metadata", None)) or {}
            preview = " ".join((meta.get("text") or "").split())[:100]
            self.stdout.write(f"  {raw_id}\n      {preview}\n")

        legacy = [m for m in matches if not str(
            m["id"] if isinstance(m, dict) else m.id
        ).rsplit("#", 1)[-1].isdigit()]
        if legacy:
            self.stdout.write(self.style.WARNING(
                f"{len(legacy)} chunk(s) still carry pre-migration random ids and "
                f"cannot be fused with sparse results. Run `manage.py reindex_hybrid`."
            ))
