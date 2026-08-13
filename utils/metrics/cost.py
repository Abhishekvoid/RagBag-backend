import logging
from threading import Lock
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class CostTracker:
    """Per-token spend accounting for every LLM call the app makes.

    A model missing from PRICING is NOT billed at zero — its tokens land in a
    separate `unpriced` bucket and log a warning, so a stale price table shows
    up as an explicit gap instead of a silently shrinking bill.
    """

    # USD per token. Verify at the provider's pricing page before quoting these
    # numbers anywhere they matter.
    #
    # The OpenRouter entries are priced at a genuine zero (`:free` variants),
    # which is deliberately different from being absent: absent means "the table
    # is stale" and routes tokens to the `unpriced` bucket, whereas zero means
    # "we know this call is free". Drop the `:free` suffix and these numbers
    # stop being true.
    PRICING = {
        # Routing / query expansion / follow-ups / all JSON-mode calls
        "nvidia/nemotron-3-super-120b-a12b:free": {"input": 0.0, "output": 0.0},
        # Answer generation
        "nvidia/nemotron-3-ultra-550b-a55b:free": {"input": 0.0, "output": 0.0},
    }

    def __init__(self):
        self._lock = Lock()
        self._reset_counters()
        self._day = datetime.now().date()

    def _reset_counters(self):
        self._daily_cost = 0.0
        self._token_usage = {"input": 0, "output": 0}
        self._requests = 0
        self._unpriced = {}

    def _roll_day_if_needed(self):
        """Caller must hold the lock. Keeps 'daily' honest — without this the
        counters accumulate for the whole life of the process."""
        today = datetime.now().date()
        if today != self._day:
            self._reset_counters()
            self._day = today

    def track(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> Dict[str, float]:
        """Record one API call and return its cost."""
        pricing = self.PRICING.get(model)

        if pricing is None:
            logger.warning("no pricing entry for model %r — tokens counted as unpriced", model)
            with self._lock:
                self._roll_day_if_needed()
                self._requests += 1
                self._token_usage["input"] += input_tokens
                self._token_usage["output"] += output_tokens
                self._unpriced[model] = self._unpriced.get(model, 0) + input_tokens + output_tokens
                daily_total = self._daily_cost
            return {
                "cost_usd": 0.0,
                "priced": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "daily_total": round(daily_total, 6),
                "monthly_projection": round(daily_total * 30, 4),
            }

        total_cost = input_tokens * pricing["input"] + output_tokens * pricing.get("output", 0)

        with self._lock:
            self._roll_day_if_needed()
            self._daily_cost += total_cost
            self._requests += 1
            self._token_usage["input"] += input_tokens
            self._token_usage["output"] += output_tokens
            daily_total = self._daily_cost

        return {
            "cost_usd": round(total_cost, 6),
            "priced": True,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "daily_total": round(daily_total, 4),
            "monthly_projection": round(daily_total * 30, 2),
        }

    def reset_daily(self):
        """Cron job at midnight (track() also rolls over on its own)."""
        with self._lock:
            self._reset_counters()
            self._day = datetime.now().date()

    def get_summary(self) -> Dict:
        with self._lock:
            requests = self._requests
            daily_cost = self._daily_cost
            tokens = dict(self._token_usage)
            unpriced = dict(self._unpriced)
            day = self._day.isoformat()

        return {
            "date": day,
            "requests": requests,
            # 6dp: per-query spend here is fractions of a cent, and rounding to
            # 2dp reports a busy day as $0.00.
            "daily_cost_usd": round(daily_cost, 6),
            "monthly_projection_usd": round(daily_cost * 30, 4),
            "token_usage": tokens,
            "cost_per_1k_requests_usd": round(daily_cost / requests * 1000, 4) if requests else 0.0,
            # Non-empty means the price table is missing a model in active use;
            # daily_cost_usd is an undercount until it is filled in.
            "unpriced_tokens_by_model": unpriced,
        }


# Production singleton
cost_tracker = CostTracker()
