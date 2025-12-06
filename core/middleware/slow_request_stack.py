import time,sys, threading, traceback, logging
logger = logging.getLogger("core")

THRESH_MS = 3000


class SlowRequestStackMiddelware:
    def __init__(self, get_response):
        self.get_reponse = get_response

    def __call__(self, request):
        start = time.perf_counter()
        response = self.get_response(request)
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed >= THRESH_MS:
            logger.warning("Slow request %s %s took %.2fms — dumping main thread stack", request.method, request.path, elapsed)
            for thread_id, frame in sys._current_frames().items():
                if thread_id == threading.main_thread().ident:
                    stack = ''.join(traceback.format_stack(frame))
                    logger.warning("Main thread stack:\n%s", stack)
        return response