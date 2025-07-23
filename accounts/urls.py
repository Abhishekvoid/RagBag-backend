from django.urls import path
from .views import RegisterAPIViews

urlpatterns = [
    path('register/', RegisterAPIViews.as_view(), name='custom-register'),
]
