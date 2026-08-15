"""Production-hardening regression tests.

Each test here maps to a blocker found in the pre-AWS audit. They exist to stop
the setting silently reverting — an absent AUTH_PASSWORD_VALIDATORS or a
LocMemCache in production fails open, with no error to notice.
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient

from accounts.ws_auth import WS_TICKET_TTL, consume_ticket, issue_ticket

User = get_user_model()

VALID_PASSWORD = "correct-horse-battery"


class PasswordPolicyTests(TestCase):
    """AUTH_PASSWORD_VALIDATORS was absent, so validate_password('1') passed."""

    def test_validators_are_configured(self):
        from django.conf import settings

        self.assertTrue(
            settings.AUTH_PASSWORD_VALIDATORS,
            "AUTH_PASSWORD_VALIDATORS is empty — every password would be accepted",
        )

    def test_extremely_weak_password_rejected(self):
        with self.assertRaises(ValidationError):
            validate_password("1")

    def test_short_password_rejected(self):
        with self.assertRaises(ValidationError):
            validate_password("Ab1!xyz")  # 7 chars, under the 10 minimum

    def test_common_password_rejected(self):
        with self.assertRaises(ValidationError):
            validate_password("password123")

    def test_all_numeric_password_rejected(self):
        with self.assertRaises(ValidationError):
            validate_password("9182736450")

    def test_valid_password_accepted(self):
        validate_password(VALID_PASSWORD)  # must not raise

    def test_registration_rejects_weak_password(self):
        response = APIClient().post(
            reverse("custom-register"),
            {
                "email": "weak@example.com",
                "name": "Weak",
                "password1": "1",
                "password2": "1",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(User.objects.filter(email="weak@example.com").exists())

    def test_registration_accepts_strong_password(self):
        response = APIClient().post(
            reverse("custom-register"),
            {
                "email": "strong@example.com",
                "name": "Strong",
                "password1": VALID_PASSWORD,
                "password2": VALID_PASSWORD,
            },
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(User.objects.filter(email="strong@example.com").exists())

    def test_existing_authentication_still_works(self):
        """Validators run on set, not on check — existing logins must survive."""
        User.objects.create_user(email="old@example.com", password=VALID_PASSWORD)
        response = APIClient().post(
            "/auth/jwt/create/",
            {"email": "old@example.com", "password": VALID_PASSWORD},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access", response.data)


class CacheBackendTests(TestCase):
    """DRF throttles are useless if each process keeps its own counters."""

    def test_production_cache_resolves_to_redis(self):
        import importlib
        import os

        import core.settings

        env = {
            "DEBUG": "False",
            "SECRET_KEY": "test-only-not-a-real-secret",
            "REDIS_URL": "redis://redis.internal:6379/0",
            "DJANGO_ALLOWED_HOSTS": "api.example.com",
            "CORS_ALLOWED_ORIGINS": "https://app.example.com",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
            core.settings.sys, "argv", ["manage.py", "runserver"]
        ):
            reloaded = importlib.reload(core.settings)

            self.assertEqual(
                reloaded.CACHES["default"]["BACKEND"],
                "django.core.cache.backends.redis.RedisCache",
            )
            self.assertEqual(
                reloaded.CACHES["default"]["LOCATION"], "redis://redis.internal:6379/0"
            )
            # Production must not implicitly trust a developer's machine.
            self.assertNotIn("http://localhost:3000", reloaded.CORS_ALLOWED_ORIGINS)
            # The browsable HTML API must not ship to the public internet.
            self.assertNotIn(
                "rest_framework.renderers.BrowsableAPIRenderer",
                reloaded.REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"],
            )
            # Probe paths must not be answered with a 301.
            self.assertIn(r"^healthz/?$", reloaded.SECURE_REDIRECT_EXEMPT)
            self.assertIn(r"^ping/?$", reloaded.SECURE_REDIRECT_EXEMPT)

        importlib.reload(core.settings)

    def test_missing_production_config_fails_loudly(self):
        """No silent fallback to an abandoned hostname."""
        import importlib
        import os

        import core.settings
        from django.core.exceptions import ImproperlyConfigured

        env = {
            "DEBUG": "False",
            "SECRET_KEY": "test-only-not-a-real-secret",
            "DJANGO_ALLOWED_HOSTS": "",
            "CORS_ALLOWED_ORIGINS": "",
            "REDIS_URL": "",
        }
        try:
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
                core.settings.sys, "argv", ["manage.py", "runserver"]
            ):
                with self.assertRaises(ImproperlyConfigured):
                    importlib.reload(core.settings)
        finally:
            importlib.reload(core.settings)


class DevelopmentDefaultsTests(TestCase):
    """Hardening must not make local development unusable."""

    def test_debug_keeps_localhost_and_browsable_api(self):
        from django.conf import settings

        # The suite runs with the dev .env (DEBUG=True) or TESTING=True; either
        # way localhost must be present so `npm run dev` can reach the API.
        self.assertIn("http://localhost:3000", settings.CORS_ALLOWED_ORIGINS)


class WebSocketTicketTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            email="ws@example.com", password=VALID_PASSWORD
        )
        self.client = APIClient()

    def test_ticket_requires_authentication(self):
        response = self.client.post(reverse("ws-ticket"))
        self.assertEqual(response.status_code, 401)

    def test_authenticated_user_gets_a_ticket(self):
        self.client.force_authenticate(self.user)
        response = self.client.post(reverse("ws-ticket"))

        self.assertEqual(response.status_code, 201)
        self.assertIn("ticket", response.data)
        self.assertEqual(response.data["expires_in"], WS_TICKET_TTL)
        # The ticket must be opaque, not a token carrying user data.
        self.assertNotIn(str(self.user.pk), response.data["ticket"])

    def test_ticket_resolves_to_its_user(self):
        ticket = issue_ticket(self.user)
        self.assertEqual(consume_ticket(ticket), str(self.user.pk))

    def test_ticket_is_single_use(self):
        ticket = issue_ticket(self.user)
        self.assertIsNotNone(consume_ticket(ticket))
        self.assertIsNone(
            consume_ticket(ticket), "a replayed ticket must not open a second socket"
        )

    def test_unknown_ticket_is_rejected(self):
        self.assertIsNone(consume_ticket("not-a-real-ticket"))

    def test_empty_ticket_is_rejected(self):
        self.assertIsNone(consume_ticket(""))
        self.assertIsNone(consume_ticket(None))

    def test_expired_ticket_is_rejected(self):
        """Expiry is a cache TTL; simulate it by dropping the key."""
        ticket = issue_ticket(self.user)
        cache.clear()
        self.assertIsNone(consume_ticket(ticket))


class WebSocketConnectionTests(TestCase):
    """End-to-end through the Channels middleware + consumer."""

    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user(
            email="sock@example.com", password=VALID_PASSWORD
        )

    async def _connect(self, query):
        """Drive the real ASGI app through a handshake; True if accepted.

        Hand-rolled rather than channels.testing.WebsocketCommunicator, whose
        package __init__ imports ChannelsLiveServerTestCase and therefore needs
        daphne — a server this project neither runs nor ships.
        """
        import asyncio
        from urllib.parse import urlencode, urlparse

        from core.asgi import application

        parsed = urlparse(f"/ws/notifications/{query}")
        scope = {
            "type": "websocket",
            "path": parsed.path,
            "raw_path": parsed.path.encode(),
            "query_string": parsed.query.encode(),
            "headers": [],
            "subprotocols": [],
        }

        incoming = asyncio.Queue()
        await incoming.put({"type": "websocket.connect"})
        sent = []

        async def receive():
            return await incoming.get()

        async def send(message):
            sent.append(message)

        try:
            await asyncio.wait_for(application(scope, receive, send), timeout=5)
        except asyncio.TimeoutError:
            # An accepted socket stays open waiting for the next client message.
            pass

        return any(m["type"] == "websocket.accept" for m in sent)

    def test_valid_ticket_connects(self):
        from asgiref.sync import async_to_sync

        ticket = issue_ticket(self.user)
        self.assertTrue(async_to_sync(self._connect)(f"?ticket={ticket}"))

    def test_no_ticket_is_rejected(self):
        from asgiref.sync import async_to_sync

        self.assertFalse(async_to_sync(self._connect)(""))

    def test_invalid_ticket_is_rejected(self):
        from asgiref.sync import async_to_sync

        self.assertFalse(async_to_sync(self._connect)("?ticket=bogus"))

    def test_jwt_in_query_string_is_no_longer_accepted(self):
        """The old ?token=<JWT> mechanism must be gone, not merely deprecated."""
        from asgiref.sync import async_to_sync
        from rest_framework_simplejwt.tokens import AccessToken

        token = str(AccessToken.for_user(self.user))
        self.assertFalse(async_to_sync(self._connect)(f"?token={token}"))

    def test_reconnect_needs_a_fresh_ticket(self):
        from asgiref.sync import async_to_sync

        ticket = issue_ticket(self.user)
        self.assertTrue(async_to_sync(self._connect)(f"?ticket={ticket}"))
        # Same ticket again: consumed, so the reconnect must fail.
        self.assertFalse(async_to_sync(self._connect)(f"?ticket={ticket}"))
        # A fresh ticket restores the connection.
        self.assertTrue(
            async_to_sync(self._connect)(f"?ticket={issue_ticket(self.user)}")
        )
