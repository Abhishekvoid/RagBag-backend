from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import TokenBlacklistView

from .health import healthz, ping

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
    path('auth/', include('accounts.urls')),
    path("auth/jwt/blacklist/", TokenBlacklistView.as_view(), name="token_blacklist"),
    # Liveness (shallow) and readiness (deep). Both are exempt from the HTTPS
    # redirect in production so a load-balancer probe never receives a 301.
    path("ping/", ping, name="ping"),
    path("healthz/", healthz, name="healthz"),
]
