import sys
import logging

class ColoredStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stdout
        super().__init__(stream=stream)
        self.stream.reconfigure(encoding='utf-8')