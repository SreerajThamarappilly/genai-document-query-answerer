# src/logger.py
"""
Simple logging helper.  
Configures log level and formatting once, then returns loggers per module.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level: int = getattr(logging, level_name, logging.DEBUG)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
