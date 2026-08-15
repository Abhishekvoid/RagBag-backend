import os
from django.core.asgi import get_asgi_application


django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from accounts.middleware import TicketAuthMiddleware
import accounts.routing
import os



os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')


application = ProtocolTypeRouter({
  "http": django_asgi_app,
  "websocket": TicketAuthMiddleware(
        URLRouter(
            accounts.routing.websocket_urlpatterns
        )
    ),
})
