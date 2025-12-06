import logging 
import sys

_COLOR_MAP = {
    'DEBUG': '\033[94m',    # blue
    'INFO': '\033[92m',     # green
    'WARNING': '\033[93m',  # yellow
    'ERROR': '\033[91m',    # red
    'CRITICAL': '\033[95m', # magenta
}
_RESET = '\033[0m'

class ColoredStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)

    def format(self, record):
        levelname = record.levelname
        color = _COLOR_MAP.get(levelname, '')
        record.msg = f"{color}{record.msg}{_RESET}"
        return super().format(record)