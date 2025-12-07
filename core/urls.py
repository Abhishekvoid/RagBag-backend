from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def ping(request):
    return JsonResponse({"status": "ok"})


@csrf_exempt
def debug_login(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body.decode() or "{}")
    except Exception:
        data = {}

    username = data.get("username")
    password = data.get("password")

    # No DB, no hashing, no JWT library, just a fake response
    return JsonResponse({
        "access": "dummy-access-token",
        "refresh": "dummy-refresh-token",
        "username": username,
    })
urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('djoser.urls')),            
    path('auth/', include('djoser.urls.jwt')),        
    path('auth/', include('accounts.urls')),  
    path("ping/", ping),       
     path("auth/jwt/debug/", debug_login),  
]