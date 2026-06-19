from django.test import TestCase
from rest_framework.test import APIClient
from django.contrib.auth import get_user_model

User = get_user_model()


class JWTRotationBlacklistTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.password = "supersecret123"
        self.user = User.objects.create_user(
            email="a@b.com", password=self.password, name="Tester"
        )

    def _login(self):
        res = self.client.post(
            "/auth/jwt/create/",
            {"email": "a@b.com", "password": self.password},
            format="json",
        )
        self.assertEqual(res.status_code, 200, res.content)
        return res.data["access"], res.data["refresh"]

    def test_refresh_rotates_and_blacklists_old(self):
        _, refresh = self._login()
        res = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertIn("access", res.data)
        self.assertIn("refresh", res.data)  # rotation returns a new refresh
        self.assertNotEqual(res.data["refresh"], refresh)
        # old refresh is now blacklisted -> reuse fails
        reuse = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(reuse.status_code, 401, reuse.content)

    def test_blacklist_endpoint_revokes_refresh(self):
        _, refresh = self._login()
        res = self.client.post("/auth/jwt/blacklist/", {"refresh": refresh}, format="json")
        self.assertEqual(res.status_code, 200, res.content)
        after = self.client.post("/auth/jwt/refresh/", {"refresh": refresh}, format="json")
        self.assertEqual(after.status_code, 401, after.content)
