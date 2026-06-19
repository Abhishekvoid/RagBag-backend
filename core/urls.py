from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from rest_framework_simplejwt.views import TokenBlacklistView

def ping(request):
    return JsonResponse({"status": "ok"})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('djoser.urls')),
    path('auth/', include('djoser.urls.jwt')),
    path('auth/', include('accounts.urls')),
    path("auth/jwt/blacklist/", TokenBlacklistView.as_view(), name="token_blacklist"),
    path("ping/", ping),
]
