"""
Logging utilities: sets up consistent formatting across all training scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "rlhf",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a logger with console (and optional file) handlers.

    Args:
        name:     Logger name (use module __name__ for per-module loggers).
        level:    Logging level string: DEBUG | INFO | WARNING | ERROR.
        log_file: Optional path to write logs to a file in addition to stdout.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
