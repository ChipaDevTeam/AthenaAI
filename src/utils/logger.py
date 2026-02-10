import logging
import sys

def setup_logger(name: str = "AIBot", level: int = logging.INFO) -> logging.Logger:
    LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(level=level, format=LOG_FMT, datefmt="%H:%M:%S")
    return logging.getLogger(name)

log = setup_logger()
