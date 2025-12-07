import time, logging
from django.utils.deprecation import MiddlewareMixin
logger = logging.getLogger("core.reqtimer")

class RequestTimerMiddleware(MiddlewareMixin):
    def process_request(self, request):
        request._start_time = time.perf_counter()

    def process_response(self, request, response):
        if hasattr(request, "_start_time"):
            ms = (time.perf_counter() - request._start_time) * 1000
            logger.info("REQ %s %s %.2fms", request.method, request.path, ms)
        return response