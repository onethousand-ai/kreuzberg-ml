import logging
import sys

SIMPLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_basic_logging() -> None:
    logging.basicConfig(stream=sys.stdout, format=SIMPLE_FORMAT, level=logging.DEBUG)


def configure_score_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    simple_formatter = logging.Formatter(SIMPLE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(simple_formatter)
    logger = logging.getLogger(name)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
