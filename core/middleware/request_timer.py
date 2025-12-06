import time
import logging

logger = logging.getLogger("core")

class RequestTimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.perf_counter()
        try:
            response = self.get_response(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("%s %s %s %.2fms", request.method, request.path, getattr(request, "user", "anon"), duration_ms)