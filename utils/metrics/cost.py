from threading import Lock
from typing import Dict, Any
from datetime import datetime, timedelta

class CostTracker:
    # Production pricing (2026 rates)
    PRICING = {
        "tei-embedding-3-small": {"input": 0.00002 / 1000},
        "grok-4": {"input": 0.0001 / 1000, "output": 0.0003 / 1000},
        "gemini-1.5-pro": {"input": 0.00025 / 1000, "output": 0.00075 / 1000},
    }
    
    def __init__(self):
        self._daily_cost = 0.0
        self._token_usage = {"input": 0, "output": 0}
        self._lock = Lock()
        self._start_date = datetime.now().date()
    
    def track(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> Dict[str, float]:
        """Track single API call"""
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"] 
        total_cost = input_cost + output_cost
        
        with self._lock:
            self._daily_cost += total_cost
            self._token_usage["input"] += input_tokens
            self._token_usage["output"] += output_tokens
        
        return {
            "cost_usd": round(total_cost, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "daily_total": round(self._daily_cost, 2),
            "monthly_projection": round(self._daily_cost * 30, 2)
        }
    
    def reset_daily(self):
        """Cron job at midnight"""
        self._daily_cost = 0.0
        self._token_usage = {"input": 0, "output": 0}
    
    def get_summary(self) -> Dict:
        return {
            "daily_cost_usd": round(self._daily_cost, 2),
            "monthly_projection_usd": round(self._daily_cost * 30, 2),
            "token_usage": self._token_usage,
            "cost_per_1k_requests": round(self._daily_cost * 1000 / max(self._token_usage["input"], 1), 4)
        }

# Production singleton
cost_tracker = CostTracker()