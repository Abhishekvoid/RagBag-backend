from collections import deque
from time import perf_counter
from threading import Lock
from typing import Dict, Any, AsyncContextManager
from contextlib import asynccontextmanager

class StageLatencyTracker:
    def __init__(self, window_size: int = 1000):  # Scales to 1000+ users
        self._window_size = window_size
        self._metrics = {}
        self._lock = Lock()  # Thread-safe for multi-process
        
    def track(self, stage: str, func: callable, *args, **kwargs) -> Any:
        """Sync tracking - FastAPI, Django, Celery"""
        start = perf_counter()
        result = func(*args, **kwargs)
        latency_ms = (perf_counter() - start) * 1000
        
        with self._lock:
            if stage not in self._metrics:
                self._metrics[stage] = deque(maxlen=self.window_size)
            self._metrics[stage].append(latency_ms)
        
        return result
    
    @asynccontextmanager
    async def track_async(self, stage: str) -> AsyncContextManager:
        """Async tracking - Your RAG pipeline"""
        start = perf_counter()
        try:
            yield
        finally:
            latency_ms = (perf_counter() - start) * 1000
            with self._lock:
                if stage not in self._metrics:
                    self._metrics[stage] = deque(maxlen=self._window_size)
                self._metrics[stage].append(latency_ms)
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Production dashboard metrics"""
        metrics = {}
        with self._lock:
            for stage, latencies in self._metrics.items():
                if latencies:
                    metrics[stage] = {
                        "count": len(latencies),
                        "avg_ms": sum(latencies) / len(latencies),
                        "p95_ms": sorted(latencies)[-int(len(latencies) * 0.95)],
                        "p99_ms": sorted(latencies)[-int(len(latencies) * 0.99)] if len(latencies) > 100 else 0
                    }
        return metrics

# Global production instance (singleton pattern)
latency_tracker = StageLatencyTracker(window_size=5000)