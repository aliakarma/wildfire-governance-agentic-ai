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
    return _KwargsLoggerAdapter(logger)


class _KwargsLoggerAdapter:
    """Give a stdlib ``Logger`` the structlog keyword-argument call style.

    Call sites throughout the package log in structlog form::

        logger.info("event_name", key=value, other=value)

    ``logging.Logger`` rejects arbitrary keyword arguments with a ``TypeError``,
    so without this adapter every such call raises when ``structlog`` is absent.
    That failure mode is not cosmetic: it took down
    ``GovernanceSmartContract.attempt_unauthorised_injection`` — which logs on
    the blocking path — turning a successful block into an exception and
    breaking the Theorem 1 test suite on any install without ``structlog``.

    Keyword arguments are appended to the message as ``key=value`` pairs.
    Standard logging kwargs (``exc_info``, ``stack_info``, ``stacklevel``,
    ``extra``) are passed through to the underlying logger untouched.
    """

    _PASSTHROUGH = ("exc_info", "stack_info", "stacklevel", "extra")

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __getattr__(self, item: str) -> Any:
        # Expose the underlying logger's other attributes (level, handlers, ...).
        return getattr(self._logger, item)

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        passthrough = {k: kwargs.pop(k) for k in self._PASSTHROUGH if k in kwargs}
        if kwargs:
            rendered = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
            event = f"{event} {rendered}"
        self._logger.log(level, event, **passthrough)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, event, **kwargs)
