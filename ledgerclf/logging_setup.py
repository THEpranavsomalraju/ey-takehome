"""Logging configuration with an ASCII-safe formatter.

Windows consoles often use cp1252. Escaping non-ASCII characters keeps every
log line printable on any console (docs/05 section 5.5).
"""

from __future__ import annotations

import logging

_HANDLER_NAME = "ledgerclf-console"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class AsciiFormatter(logging.Formatter):
    """Formatter which escapes every non-ASCII character."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, then escape non-ASCII characters."""
        text = super().format(record)
        return text.encode("ascii", "backslashreplace").decode("ascii")


def configure(level: int | str = logging.INFO) -> None:
    """Configure the package logger with one ASCII-safe stderr handler.

    Calling this twice replaces the handler instead of stacking duplicates.

    Args:
        level: Logging level for the ``ledgerclf`` logger.
    """
    logger = logging.getLogger("ledgerclf")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(AsciiFormatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
