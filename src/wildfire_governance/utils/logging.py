"""Structured logging via structlog with rich console fallback."""
from __future__ import annotations

import logging
import sys
from typing import Any

try:
    import structlog
    _STRUCTLOG = True
except ImportError:
    _STRUCTLOG = False


def get_structured_logger(name: str) -> Any:
    """Return a structured logger bound to *name*.

    Uses ``structlog`` if installed, otherwise returns a standard
    ``logging.Logger``. Both expose the same ``info``, ``warning``,
    ``error``, and ``debug`` interface.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        A structlog bound logger or a standard Python logger.
    """
    if _STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(name)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
