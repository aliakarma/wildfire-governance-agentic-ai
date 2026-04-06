"""Structured logging via structlog with fallback."""
from __future__ import annotations
import logging, sys
from typing import Any

try:
    import structlog; _STRUCTLOG = True
except ImportError:
    _STRUCTLOG = False

def get_structured_logger(name: str) -> Any:
    if _STRUCTLOG:
        structlog.configure(processors=[structlog.stdlib.add_log_level, structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
            logger_factory=structlog.stdlib.LoggerFactory(), wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True)
        return structlog.get_logger(name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO); return logger
