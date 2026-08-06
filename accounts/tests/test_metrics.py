from rest_framework.test import APITestCase

from accounts.models import CustomUserModel
from utils.metrics.cost import CostTracker


class MetricsEndpointTest(APITestCase):
    def setUp(self):
        self.staff = CustomUserModel.objects.create_user(
            email="staff@b.com", password="x", name="S", is_staff=True
        )
        self.normal = CustomUserModel.objects.create_user(
            email="user@b.com", password="x", name="U"
        )

    def test_staff_gets_all_tracker_sections(self):
        self.client.force_authenticate(self.staff)
        r = self.client.get("/auth/metrics/")
        self.assertEqual(r.status_code, 200)
        for key in ("latency_ms_by_stage", "cost", "retrieval", "circuit_breakers"):
            self.assertIn(key, r.data)

    def test_non_staff_forbidden(self):
        self.client.force_authenticate(self.normal)
        self.assertEqual(self.client.get("/auth/metrics/").status_code, 403)

    def test_anonymous_forbidden(self):
        self.assertEqual(self.client.get("/auth/metrics/").status_code, 401)


class CostTrackerTest(APITestCase):
    def test_answer_model_is_priced(self):
        """Regression: llama-3.3-70b-versatile was missing from PRICING, so
        every answer call was booked at $0.00."""
        t = CostTracker()
        out = t.track("llama-3.3-70b-versatile", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertTrue(out["priced"])
        self.assertAlmostEqual(out["cost_usd"], 0.59 + 0.79, places=4)

    def test_unknown_model_is_flagged_not_zero_billed(self):
        t = CostTracker()
        out = t.track("some-new-model", input_tokens=100, output_tokens=50)
        self.assertFalse(out["priced"])
        self.assertEqual(t.get_summary()["unpriced_tokens_by_model"], {"some-new-model": 150})

    def test_summary_divides_by_requests_not_tokens(self):
        t = CostTracker()
        for _ in range(4):
            t.track("llama-3.1-8b-instant", input_tokens=1000, output_tokens=500)
        s = t.get_summary()
        # 4 calls x (1000 in @ $0.05/M + 500 out @ $0.08/M) = $0.00036
        per_call = 1000 * 0.05 / 1_000_000 + 500 * 0.08 / 1_000_000
        self.assertEqual(s["requests"], 4)
        self.assertAlmostEqual(s["daily_cost_usd"], per_call * 4, places=6)
        self.assertAlmostEqual(s["cost_per_1k_requests_usd"], per_call * 1000, places=4)

    def test_empty_summary_does_not_divide_by_zero(self):
        self.assertEqual(CostTracker().get_summary()["cost_per_1k_requests_usd"], 0.0)
