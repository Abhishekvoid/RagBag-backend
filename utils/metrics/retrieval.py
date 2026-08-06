from typing import List, Dict
from threading import Lock
from dataclasses import dataclass

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