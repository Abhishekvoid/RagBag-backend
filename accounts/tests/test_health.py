"""Liveness and readiness probe tests.

/ping/ must stay shallow (a load balancer polls it constantly) and /healthz must
distinguish a hard dependency outage from a degraded-but-serving state.
"""

from unittest import mock

from django.test import TestCase, override_settings


class HealthEndpointTests(TestCase):
    def test_ping_is_shallow_and_unauthenticated(self):
        with mock.patch("core.health._check_postgres") as db_check:
            response = self.client.get("/ping/")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok"})
            db_check.assert_not_called()

    @override_settings(AWS_STORAGE_BUCKET_NAME=None)
    def test_healthz_ok_when_hard_deps_up(self):
        with mock.patch("core.health._check_redis", return_value={"status": "ok"}), \
             mock.patch("core.health._check_celery", return_value={"status": "ok"}), \
             mock.patch("core.health._check_tei", return_value={"status": "ok"}):
            response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["hard"]["postgres"]["status"], "ok")

    def test_healthz_503_when_postgres_down(self):
        with mock.patch(
            "core.health._check_postgres", side_effect=RuntimeError("boom")
        ), mock.patch("core.health._check_redis", return_value={"status": "ok"}):
            response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "error")

    def test_healthz_503_when_redis_down(self):
        with mock.patch(
            "core.health._check_redis", side_effect=RuntimeError("boom")
        ):
            response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 503)

    def test_healthz_200_degraded_when_soft_dep_down(self):
        with mock.patch("core.health._check_redis", return_value={"status": "ok"}), \
             mock.patch("core.health._check_celery", side_effect=RuntimeError("boom")), \
             mock.patch("core.health._check_s3", return_value={"status": "ok"}), \
             mock.patch("core.health._check_tei", return_value={"status": "ok"}):
            response = self.client.get("/healthz/")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "degraded")
        self.assertIn("celery", body["degraded"])

    def test_healthz_never_leaks_credentials(self):
        """Connection errors embed host/user/password — only the class name ships."""
        secret = "p@ssw0rd-should-never-appear"
        with mock.patch(
            "core.health._check_redis",
            side_effect=RuntimeError(f"auth failed for redis://user:{secret}@host"),
        ):
            response = self.client.get("/healthz/")

        self.assertNotIn(secret, response.content.decode())
        self.assertEqual(response.json()["hard"]["redis"]["error"], "RuntimeError")
