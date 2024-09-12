# from transformers import logging as hf_logging
# logger = hf_logging.get_logger(__name__)
# logger.warning("testsdfs")

from loggers.logging_colors import get_logger

logger = get_logger(__name__, "debug.log")
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")