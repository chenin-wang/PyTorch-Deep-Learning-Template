import logging
import platform
import logging.handlers
import os
import ctypes
from typing import Optional
from transformers.utils import logging as hf_logging

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
        'RESET': '\033[0m'      # Reset color
    }

    def format(self, record):
        log_message = super().format(record)
        if platform.system() != 'Windows':
            return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"
        return log_message

def get_logger(log_path: Optional[str] = None):
    console_formatter = ColoredFormatter("%(asctime)s - %(module)s: [%(levelname)8s] - %(message)s", 
                                         datefmt='%Y-%m-%d %H:%M:%S')

    hf_logger = hf_logging.get_logger("transformers")
    all_handlers = hf_logger.handlers

    # Print information about each handler
    for i, handler in enumerate(all_handlers):
        if handler and isinstance(hf_logger.handlers[i], logging.StreamHandler):
            handler.setFormatter(console_formatter)
        else:
            print("Warning: No handlers found for the logger.")
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            hf_logger.addHandler(console_handler)

    # Set up file handler
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(module)s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        hf_logger.addHandler(file_handler)

    return hf_logger

def add_coloring_to_emit_windows(fn):
    def _set_color(self, code):
        STD_OUTPUT_HANDLE = -11
        hdl = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        ctypes.windll.kernel32.SetConsoleTextAttribute(hdl, code)

    setattr(logging.StreamHandler, '_set_color', _set_color)

    def new(*args):
        FOREGROUND_BLUE = 0x0001
        FOREGROUND_GREEN = 0x0002
        FOREGROUND_RED = 0x0004
        FOREGROUND_INTENSITY = 0x0008
        FOREGROUND_WHITE = FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED

        colors = {
            logging.DEBUG: FOREGROUND_BLUE | FOREGROUND_INTENSITY,
            logging.INFO: FOREGROUND_GREEN,
            logging.WARNING: FOREGROUND_RED | FOREGROUND_GREEN,
            logging.ERROR: FOREGROUND_RED,
            logging.CRITICAL: FOREGROUND_RED | FOREGROUND_INTENSITY
        }

        color = colors.get(args[1].levelno, FOREGROUND_WHITE)

        args[0]._set_color(color)
        ret = fn(*args)
        args[0]._set_color(FOREGROUND_WHITE)
        return ret
    return new

if platform.system() == 'Windows':
    logging.StreamHandler.emit = add_coloring_to_emit_windows(logging.StreamHandler.emit)
else:
    os.system('')  # Enable ANSI colors for Windows 10+

if __name__ == "__main__":
    hf_logging.set_verbosity_info()
    logger = get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
