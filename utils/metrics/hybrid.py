"""Counters for the hybrid (dense + sparse) retrieval path.

These exist because hybrid retrieval degrades SILENTLY by design. When the
sparse half is unavailable, or when a chapter was indexed before the sparse
index existed, RRF simply fuses fewer lists and the user still gets an answer.
That is the correct behaviour — a query must never fail because a second index
is missing — but without a counter it is indistinguishable from working.

The distinction that matters is between the two ways the sparse half can
contribute nothing, because they need opposite responses:

  sparse_unavailable  the service errored, timed out, or the breaker is open.
                      An operational problem. Look at Pinecone.

  sparse_empty        the service answered correctly and returned no matches
                      for this chapter. Almost always means the chapter has not
                      been through `manage.py reindex_hybrid` yet. A backfill
                      problem, not an outage.

Same observable symptom (dense-only results), completely different fix. Rolling
them into one "hybrid failed" counter would hide exactly the thing worth
knowing, so they are counted separately.

Per-process and in-memory, with the same caveat as every other tracker here:
these numbers describe the worker that served the request, not the fleet.
"""

from threading import Lock


class HybridRetrievalStats:
    def __init__(self):
        self._lock = Lock()
        self._reset()

    def _reset(self):
        self._queries = 0
        self._hybrid_ok = 0
        self._sparse_unavailable = 0
        self._sparse_empty = 0
        self._disabled = 0

    def record(self, *, outcome: str):
        """Record one retrieval. `outcome` is one of the four states above.

        Unknown outcomes are counted toward the query total but nothing else,
        so a typo at a call site shows up as an accounting gap rather than
        silently inflating a bucket that someone is paging on.
        """
        with self._lock:
            self._queries += 1
            if outcome == "hybrid_ok":
                self._hybrid_ok += 1
            elif outcome == "sparse_unavailable":
                self._sparse_unavailable += 1
            elif outcome == "sparse_empty":
                self._sparse_empty += 1
            elif outcome == "disabled":
                self._disabled += 1

    def get_summary(self) -> dict:
        with self._lock:
            total = self._queries
            hybrid_ok = self._hybrid_ok
            unavailable = self._sparse_unavailable
            empty = self._sparse_empty
            disabled = self._disabled

        def rate(n):
            return round(n / total, 4) if total else 0.0

        return {
            "queries": total,
            "hybrid_ok": hybrid_ok,
            "sparse_unavailable": unavailable,
            "sparse_empty": empty,
            "disabled": disabled,
            # The headline number: how often retrieval ran on dense alone.
            "dense_only_rate": rate(unavailable + empty + disabled),
        }

    def reset(self):
        with self._lock:
            self._reset()


hybrid_stats = HybridRetrievalStats()
