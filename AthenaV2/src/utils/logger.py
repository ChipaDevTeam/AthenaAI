import logging

def setup_logger(name="AIBot", level=logging.INFO):
    LOG_FMT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(level=level, format=LOG_FMT, datefmt="%H:%M:%S")
    return logging.getLogger(name)

log = setup_logger()
