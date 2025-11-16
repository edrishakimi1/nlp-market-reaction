"""Consistent logging configuration for the project."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logging(config: Any) -> None:
    """Configure root logging handlers using settings from *config*."""
    log_cfg = getattr(config, "logging", None) or {}
    level_name = getattr(log_cfg, "level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    log_file = getattr(log_cfg, "log_file", None)

    handlers: list[logging.Handler] = [
        logging.StreamHandler()
    ]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=handlers)
