"""
TerraAlert — Metadata management module.

Provides atomic, merge-safe updates to outputs/meta.json.
Guarantees that existing keys are never wiped out when one pipeline component
updates its status or fallbacks.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Union

from backend.common.log import get_logger

logger = get_logger("meta")

_UNSET = object()


def _set_dotted_key(target: dict[str, Any], dotted_key: str, val: Any) -> None:
    """Set a value in target using dotted notation, e.g. 'fallbacks.confidence'."""
    keys = dotted_key.split(".")
    curr = target
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]

    last = keys[-1]
    if isinstance(val, dict) and isinstance(curr.get(last), dict):
        _deep_merge_dict(curr[last], val)
    else:
        curr[last] = val


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay dict into base dict."""
    for key, val in overlay.items():
        if "." in key:
            _set_dotted_key(base, key, val)
        elif isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], val)
        else:
            base[key] = val
    return base


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write dictionary to JSON atomically using a temp file in the same directory."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = path.with_name(f".{path.name}.{os.getpid()}_{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, path)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise


def load_meta(path: Union[Path, str]) -> dict[str, Any]:
    """Load metadata from JSON file if exists, returning empty dict if missing or corrupt."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        return json.loads(content)
    except Exception as exc:
        logger.warning("Could not read existing metadata from %s: %s", p, exc)
        return {}


def update_meta(
    path: Union[Path, str],
    key_or_data: Any,
    value: Any = _UNSET,
) -> dict[str, Any]:
    """
    Atomically updates a metadata JSON file by merging keys.
    Preserves all existing keys not being modified.

    Usage:
      update_meta(path, {"fallbacks.confidence": "not run yet"})
      update_meta(path, "fallbacks.confidence", "not run yet")
      update_meta(path, {"fallbacks": {"confidence": "not run yet"}})
    """
    p = Path(path)
    current = load_meta(p)

    if value is not _UNSET:
        key = str(key_or_data)
        if "." in key:
            _set_dotted_key(current, key, value)
        elif isinstance(value, dict) and isinstance(current.get(key), dict):
            _deep_merge_dict(current[key], value)
        else:
            current[key] = value
    elif isinstance(key_or_data, dict):
        _deep_merge_dict(current, key_or_data)
    else:
        raise ValueError(
            f"Expected dict or (key, value) pair, got {type(key_or_data).__name__}"
        )

    _atomic_write_json(p, current)
    return current
