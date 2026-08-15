# in accounts/middleware.py

from urllib.parse import parse_qs

from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

from .ws_auth import consume_ticket

User = get_user_model()


@database_sync_to_async
def get_user_from_ticket(ticket):
    """Redeem a WebSocket ticket for its user, or AnonymousUser."""
    user_id = consume_ticket(ticket)
    if user_id is None:
        return AnonymousUser()

    try:
        return User.objects.get(pk=user_id)
    except (User.DoesNotExist, ValueError, TypeError):
        return AnonymousUser()


class TicketAuthMiddleware(BaseMiddleware):
    """Authenticate a socket from a short-lived ticket in the query string.

    The long-lived JWT is deliberately NOT accepted here — see accounts/ws_auth.py
    for why a credential in a URL has to be single-use and short-lived.
    """

    async def __call__(self, scope, receive, send):
        query_string = scope.get("query_string", b"").decode("utf-8")
        params = parse_qs(query_string)
        ticket = params.get("ticket", [None])[0]

        scope["user"] = await get_user_from_ticket(ticket)

        return await super().__call__(scope, receive, send)
