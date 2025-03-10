import logging
from dotenv import load_dotenv
import os
import openai

load_dotenv()

def get_logger(logger_name: str, log_level=logging.DEBUG):
    """Creates and configures a logger with console and optional file logging."""

    # Configure OpenAI and third-party logging suppression
    #openai.log = "warn"
    suppressed_loggers = [
        "pymongo", "_trace", "httpx", "httpx._trace", "httpcore", "openai",
        "openai._trace", "openai._client", "openai.http", "urllib3", "asyncio",
        "nltk_data", "transformers", "sentencepiece", "sentence_transformers"
    ]

    for log in suppressed_loggers:
        logging.getLogger(log).setLevel(logging.WARNING)

    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent log propagation

    if not logger.handlers:  # Avoid duplicate handlers on multiple calls
        log_format = "%(asctime)s [%(process)d] %(levelname)s %(filename)s : %(lineno)d - %(message)s"
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (if logging to file is enabled)
        file_name = os.getenv("backend_log_file")
        if os.getenv("generate_debug_logs") == "true" and file_name:
            file_handler = logging.FileHandler(file_name, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
