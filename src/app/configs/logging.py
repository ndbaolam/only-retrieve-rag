# app/configs/logging.py
import logging

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
}

def setup_logging():
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"]
    )
