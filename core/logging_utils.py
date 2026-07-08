"""A tiny logging helper so every module logs in a consistent format.

Named ``logging_utils`` rather than ``logging`` to avoid shadowing the stdlib
module within the package.
"""

from __future__ import annotations

import logging

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
        _CONFIGURED = True
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
