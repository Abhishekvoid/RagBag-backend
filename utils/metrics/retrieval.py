from typing import Iterable, List, Dict, Sequence
from threading import Lock
from dataclasses import dataclass


# --- offline evaluation ------------------------------------------------------
#
# These are for the golden-set harness (manage.py eval_retrieval), NOT for the
# live request path.
#
# They exist because the online metric below cannot be used to judge hybrid
# retrieval. `RetrievalEvaluator.evaluate` scores relevance by keyword overlap
# between the query and the chunk; hybrid retrieval ADDS a lexical retriever.
# Measuring a lexical retriever with a lexical metric is circular — the number
# would rise whether or not students got better answers, which is worse than
# having no number at all.
#
# Recall and MRR against hand-labelled relevant chunks are independent of how
# the chunks were found, which is the only property that makes a before/after
# comparison mean anything.


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k.

    Answers "did retrieval find the material?" — the question hybrid retrieval
    is supposed to improve.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & relevant) / len(relevant)


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Reciprocal rank of the FIRST relevant chunk in the top k, else 0.

    Answers "did retrieval rank the material highly?" — which is the question
    that actually matters once the candidate pool is small enough that recall
    saturates, and the question RRF is supposed to improve. Reported alongside
    recall precisely because the two can move independently.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for position, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0

@dataclass
class RetrievalMetrics:
    total_queries: int = 0
    exact_hits: int = 0
    relevant_hits: int = 0
    avg_chunks_retrieved: float = 0.0
    
class RetrievalEvaluator:
    def __init__(self):
        self._metrics = RetrievalMetrics()
        self._lock = Lock()
    
    def evaluate(self, query: str, chunks: List[str], ground_truth: str = None) -> Dict:
        """Production retrieval scoring"""
        chunks_count = len(chunks)
        
        # Exact match (production gold standard)
        exact_match = any(ground_truth and ground_truth.lower() in chunk.lower() 
                         for chunk in chunks) if ground_truth else False
        
        # Relevance score (keyword overlap)
        query_words = set(query.lower().split())
        relevant_chunks = sum(1 for chunk in chunks 
                             if len(set(chunk.lower().split()) & query_words) / len(query_words) > 0.3)
        
        with self._lock:
            self._metrics.total_queries += 1
            if exact_match:
                self._metrics.exact_hits += 1
            if relevant_chunks > 0:
                self._metrics.relevant_hits += 1
            self._metrics.avg_chunks_retrieved = (
                (self._metrics.avg_chunks_retrieved * (self._metrics.total_queries - 1) + chunks_count) 
                / self._metrics.total_queries
            )
        
        return {
            "chunks_retrieved": chunks_count,
            "exact_hit": exact_match,
            "relevance_rate": relevant_chunks / max(chunks_count, 1),
            "global_metrics": {
                "exact_hit_rate": self._metrics.exact_hits / self._metrics.total_queries,
                "relevance_rate": self._metrics.relevant_hits / self._metrics.total_queries,
                "avg_chunks": self._metrics.avg_chunks_retrieved
            }
        }

    def get_summary(self) -> Dict:
        """Rolled-up retrieval quality since process start (or last reset)."""
        with self._lock:
            m = self._metrics
            total = m.total_queries

        return {
            "total_queries": total,
            "exact_hit_rate": round(m.exact_hits / total, 4) if total else 0.0,
            "relevance_rate": round(m.relevant_hits / total, 4) if total else 0.0,
            "avg_chunks_retrieved": round(m.avg_chunks_retrieved, 2),
        }


retrieval_evaluator = RetrievalEvaluator()