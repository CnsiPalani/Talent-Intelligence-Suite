
import logging
from pathlib import Path
from .configuration import PATHS

def get_logger(name: str) -> logging.Logger:
    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # File
    fh = logging.FileHandler(PATHS.logs_dir / f"{name}.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
