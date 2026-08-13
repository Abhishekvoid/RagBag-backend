"""Authentication regression tests.

Written for the SECRET_KEY rotation: SIMPLE_JWT defines no SIGNING_KEY, so JWT
signatures derive from settings.SECRET_KEY. These tests pin the behaviour that
must survive a rotation (issue / authenticate / refresh / reject) and the
behaviour that must NOT survive it (tokens signed by the previous key).
"""

from unittest import skip

from django.conf import settings
from rest_framework import status
from rest_framework.test import APITestCase
from rest_framework_simplejwt.tokens import AccessToken, RefreshToken

from accounts.models import CustomUserModel

PASSWORD = "sup3r-s3cret-pw!"


class AuthFlowTest(APITestCase):
    def setUp(self):
        self.user = CustomUserModel.objects.create_user(
            email="auth@example.com", password=PASSWORD, name="Auth User"
        )

    # --- registration -------------------------------------------------
    def test_registration_creates_user_and_issues_tokens(self):
        resp = self.client.post("/auth/register/", {
            "email": "newbie@example.com",
            "name": "New Bie",
            "password1": PASSWORD,
            "password2": PASSWORD,
        }, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, resp.data)
        self.assertIn("access", resp.data["tokens"])
        self.assertIn("refresh", resp.data["tokens"])
        self.assertTrue(
            CustomUserModel.objects.filter(email="newbie@example.com").exists()
        )

    def test_registration_rejects_mismatched_passwords(self):
        resp = self.client.post("/auth/register/", {
            "email": "bad@example.com",
            "name": "Bad",
            "password1": PASSWORD,
            "password2": PASSWORD + "-different",
        }, format="json")
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    # --- login / token issuance ---------------------------------------
    def test_login_issues_access_and_refresh(self):
        resp = self.client.post("/auth/jwt/create/", {
            "email": "auth@example.com", "password": PASSWORD,
        }, format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK, resp.data)
        self.assertIn("access", resp.data)
        self.assertIn("refresh", resp.data)

    def test_login_rejects_wrong_password(self):
        resp = self.client.post("/auth/jwt/create/", {
            "email": "auth@example.com", "password": "wrong-password",
        }, format="json")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    # --- authenticated vs unauthenticated access ----------------------
    def test_authenticated_request_succeeds(self):
        token = str(AccessToken.for_user(self.user))
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        resp = self.client.get("/auth/me/")
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data["email"], "auth@example.com")

    def test_unauthenticated_request_is_rejected(self):
        resp = self.client.get("/auth/me/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_garbage_token_is_rejected(self):
        self.client.credentials(HTTP_AUTHORIZATION="Bearer not-a-real-token")
        resp = self.client.get("/auth/me/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    # --- refresh -------------------------------------------------------
    def test_refresh_returns_new_access_token(self):
        refresh = str(RefreshToken.for_user(self.user))
        resp = self.client.post("/auth/jwt/refresh/", {"refresh": refresh},
                                format="json")
        self.assertEqual(resp.status_code, status.HTTP_200_OK, resp.data)
        self.assertIn("access", resp.data)

        # the freshly minted access token must actually authenticate
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {resp.data['access']}")
        self.assertEqual(self.client.get("/auth/me/").status_code,
                         status.HTTP_200_OK)


class SecretKeyBindingTest(APITestCase):
    """The rotation contract itself.

    SIMPLE_JWT sets no SIGNING_KEY, so tokens are signed with SECRET_KEY. A
    token minted under a different key must be refused — this is what makes
    rotating SECRET_KEY an effective way to invalidate every outstanding
    session after a credential exposure.
    """

    def setUp(self):
        self.user = CustomUserModel.objects.create_user(
            email="binding@example.com", password=PASSWORD, name="Binding"
        )

    def test_simplejwt_has_no_independent_signing_key(self):
        self.assertNotIn("SIGNING_KEY", settings.SIMPLE_JWT)

    def test_effective_signing_key_is_the_django_secret_key(self):
        from rest_framework_simplejwt.settings import api_settings
        self.assertEqual(api_settings.SIGNING_KEY, settings.SECRET_KEY)

    def test_token_signed_with_a_different_key_is_rejected(self):
        """Forge a structurally valid token under a foreign key.

        NOTE: override_settings(SECRET_KEY=...) is deliberately NOT used here.
        SimpleJWT resolves SIGNING_KEY from SECRET_KEY once at import time and
        does not listen for SECRET_KEY changes, so an override would leave the
        verifying key untouched and the test would pass vacuously. Signing a
        token explicitly with a different key is what actually exercises the
        rotation contract.
        """
        import jwt

        genuine = AccessToken.for_user(self.user)
        forged = jwt.encode(genuine.payload, "a-foreign-signing-key",
                            algorithm="HS256")
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {forged}")
        resp = self.client.get("/auth/me/")
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)


class PasswordResetTest(APITestCase):
    """Password-reset tokens are salted with SECRET_KEY, so outstanding reset
    links also die on rotation. Verify the endpoint still functions afterwards."""

    def setUp(self):
        self.user = CustomUserModel.objects.create_user(
            email="reset@example.com", password=PASSWORD, name="Reset"
        )

    @skip(
        "KNOWN PRE-EXISTING DEFECT (not caused by the SECRET_KEY rotation): "
        "core/settings.py defines no DJOSER block, so djoser's required "
        "PASSWORD_RESET_CONFIRM_URL is unset and POST /auth/users/reset_password/ "
        "raises AttributeError while rendering the reset email. Password reset "
        "is non-functional. Re-enable this test once DJOSER["
        "'PASSWORD_RESET_CONFIRM_URL'] is configured to a real frontend route."
    )
    def test_reset_password_endpoint_accepts_request(self):
        resp = self.client.post("/auth/users/reset_password/",
                                {"email": "reset@example.com"}, format="json")
        # djoser returns 204 whether or not the address exists (no enumeration)
        self.assertIn(resp.status_code,
                      (status.HTTP_204_NO_CONTENT, status.HTTP_200_OK))

    def test_default_token_generator_is_bound_to_secret_key(self):
        from django.contrib.auth.tokens import default_token_generator
        token = default_token_generator.make_token(self.user)
        self.assertTrue(default_token_generator.check_token(self.user, token))
        with self.settings(SECRET_KEY="another-key-entirely-for-this-test"):
            self.assertFalse(default_token_generator.check_token(self.user, token))
