"""YAML based configuration utilities."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


def load_config(path: Path) -> SimpleNamespace:
    """Load YAML config at *path* and expose nested keys as attributes."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    namespace = _to_namespace(data)
    namespace._raw = data  # type: ignore[attr-defined]
    namespace.root_dir = path.parent
    return namespace


def resolve_path(base: Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` against *base* directory."""
    return (base / relative_path).expanduser().resolve()
