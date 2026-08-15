"""Short-lived, single-use tickets for authenticating WebSocket connections.

WHY NOT THE JWT: a browser WebSocket cannot set an Authorization header, so the
credential has to travel in the URL. URLs are logged — by load balancers, by
reverse proxies, by CDNs, by browser history — so putting the long-lived access
token there leaks it into places nobody audits. Instead the client trades its
JWT (over authenticated HTTPS, in a header) for an opaque ticket that:

  * carries no user data — it is random bytes, not a signed claim;
  * expires in WS_TICKET_TTL seconds, so a log leak is worthless minutes later;
  * is consumed on first use, so a replayed URL cannot open a second socket.

Tickets live in the Django cache, which in production is the shared Redis
(see CACHES in core/settings.py). That sharing is load-bearing: the web process
that issues a ticket is usually not the process that terminates the socket.
"""

import secrets

from django.core.cache import cache

# Long enough to cover a page load and a socket handshake, short enough that a
# ticket captured from an access log is dead before anyone reads the log.
WS_TICKET_TTL = 30

WS_TICKET_PREFIX = "ws-ticket:"


def _key(ticket: str) -> str:
    return f"{WS_TICKET_PREFIX}{ticket}"


def issue_ticket(user) -> str:
    """Mint a single-use ticket for `user`. Caller must already be authenticated."""
    ticket = secrets.token_urlsafe(32)
    cache.set(_key(ticket), str(user.pk), timeout=WS_TICKET_TTL)
    return ticket


def consume_ticket(ticket: str):
    """Redeem a ticket, returning the user id, or None if it is invalid.

    Deleting before returning is what makes the ticket single-use: a second
    connection attempt with the same value finds nothing.
    """
    if not ticket:
        return None

    key = _key(ticket)
    user_id = cache.get(key)
    if user_id is None:
        return None

    cache.delete(key)
    return user_id
