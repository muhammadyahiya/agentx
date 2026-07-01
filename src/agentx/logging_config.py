"""Centralized logging configuration for agentx.

Call ``setup_logging()`` once at application startup to get structured,
consistently formatted log output across all agentx modules.

All agentx loggers live under the ``agentx.*`` namespace so a single
``logging.getLogger("agentx")`` call controls the entire library.
"""
from __future__ import annotations

import logging
import logging.config
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def setup_logging(
    level: LogLevel | str = "INFO",
    fmt: str = _FMT,
    handler: logging.Handler | None = None,
    force: bool = False,
) -> None:
    """Configure the agentx logger hierarchy.

    Safe to call multiple times — subsequent calls are no-ops unless
    ``force=True``. Does *not* touch the root Python logger so it does
    not interfere with the host application's logging setup.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", …).
        fmt: Log format string. Defaults to timestamped module-aware format.
        handler: Custom handler to attach. Defaults to stderr StreamHandler.
        force: Re-configure even if already set up.
    """
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger("agentx")
    if root.handlers and not force:
        _configured = True
        return

    for h in root.handlers[:]:
        root.removeHandler(h)

    h = handler or logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter(fmt, datefmt=_DATEFMT))
    root.addHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``agentx`` namespace.

    Equivalent to ``logging.getLogger("agentx.<name>")``.  Modules within
    agentx should use ``logging.getLogger(__name__)`` directly; this helper is
    for user code that wants a named agentx-namespaced logger.
    """
    qualified = f"agentx.{name}" if not name.startswith("agentx") else name
    return logging.getLogger(qualified)
