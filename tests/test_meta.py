"""
Unit tests for backend/common/meta.py.

Verifies:
  1. Atomic write on fresh file
  2. Merging keys without overwriting existing keys
  3. Dotted nested key access (e.g. 'fallbacks.confidence')
  4. Deep merge preserving sibling nested keys
  5. Atomic file replacement leaving no temporary files
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.common.meta import load_meta, update_meta


def test_update_meta_creates_file(tmp_path: Path):
    target = tmp_path / "meta.json"
    result = update_meta(target, {"event": "Assam Flood 2022"})

    assert target.exists()
    assert result["event"] == "Assam Flood 2022"

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["event"] == "Assam Flood 2022"


def test_update_meta_preserves_existing_keys(tmp_path: Path):
    target = tmp_path / "meta.json"
    update_meta(target, {"event": "Assam Flood 2022", "satellite": "Sentinel-1"})
    update_meta(target, {"pass": "DESCENDING"})

    loaded = load_meta(target)
    assert loaded["event"] == "Assam Flood 2022"
    assert loaded["satellite"] == "Sentinel-1"
    assert loaded["pass"] == "DESCENDING"


def test_update_meta_dotted_nested_key(tmp_path: Path):
    target = tmp_path / "meta.json"
    update_meta(target, "fallbacks.confidence", "confidence module not run yet")

    loaded = load_meta(target)
    assert "fallbacks" in loaded
    assert loaded["fallbacks"]["confidence"] == "confidence module not run yet"


def test_update_meta_deep_merge_preserves_siblings(tmp_path: Path):
    target = tmp_path / "meta.json"
    # Seed initial structure
    update_meta(
        target,
        {
            "fallbacks": {
                "s2": "cloud cover too high",
                "ml": "not run",
            }
        },
    )

    # Update one sibling via dotted key
    update_meta(target, "fallbacks.confidence", "heuristic used")

    loaded = load_meta(target)
    assert loaded["fallbacks"]["s2"] == "cloud cover too high"
    assert loaded["fallbacks"]["ml"] == "not run"
    assert loaded["fallbacks"]["confidence"] == "heuristic used"


def test_update_meta_deep_merge_dict_overlay(tmp_path: Path):
    target = tmp_path / "meta.json"
    update_meta(target, {"thresholds": {"post_db_max": -18.0, "slope_max_deg": 5.0}})
    update_meta(target, {"thresholds": {"min_object_pixels": 25}})

    loaded = load_meta(target)
    assert loaded["thresholds"]["post_db_max"] == -18.0
    assert loaded["thresholds"]["slope_max_deg"] == 5.0
    assert loaded["thresholds"]["min_object_pixels"] == 25


def test_update_meta_invalid_type_raises(tmp_path: Path):
    target = tmp_path / "meta.json"
    with pytest.raises(ValueError):
        update_meta(target, 12345)


def test_no_temp_files_leftover(tmp_path: Path):
    target = tmp_path / "meta.json"
    for i in range(5):
        update_meta(target, f"run_{i}", i)

    # Check directory contents — should only have meta.json
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "meta.json"
