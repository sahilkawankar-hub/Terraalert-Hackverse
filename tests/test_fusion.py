"""
Unit tests for backend/fusion/fuse.py.

Verifies:
  1. agreement.tif encoding (0 neither, 1 one method, 2 both, 255 nodata).
  2. Single-method treatment and method_mask.tif when ML is unavailable (255).
  3. Fused flood rules: classical_plus_ml, intersection, union.
  4. run_fusion pipeline creates all rasters, verifies grid alignment, and updates meta.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import config
from backend.fusion.fuse import (
    build_agreement_raster,
    build_fused_flood_raster,
    run_fusion,
)


def test_build_agreement_raster():
    """Verify agreement and method mask encoding across all combinations."""
    # Shapes: 1 row, 5 cols
    classical = np.array([[0, 1, 1, 0, 255]], dtype=np.uint8)
    ml =        np.array([[0, 1, 0, 1, 255]], dtype=np.uint8)

    agree, mm = build_agreement_raster(classical, ml)

    # col 0: neither flags flood -> 0
    assert agree[0, 0] == 0
    assert mm[0, 0] == 0

    # col 1: both flag flood -> 2
    assert agree[0, 1] == 2
    assert mm[0, 1] == 0

    # col 2: classical flags flood, ml does not -> 1
    assert agree[0, 2] == 1
    assert mm[0, 2] == 0

    # col 3: classical does not flag flood, ml does -> 1
    assert agree[0, 3] == 1
    assert mm[0, 3] == 0

    # col 4: both nodata -> 255
    assert agree[0, 4] == 255
    assert mm[0, 4] == 255


def test_agreement_when_ml_unavailable():
    """Where ML is 255 but classical is valid, treat as single-method and set method_mask=1."""
    classical = np.array([[1, 0]], dtype=np.uint8)
    ml =        np.array([[255, 255]], dtype=np.uint8)

    agree, mm = build_agreement_raster(classical, ml)

    # Classical flags flood -> single method (1)
    assert agree[0, 0] == 1
    assert mm[0, 0] == 1

    # Classical does not flag flood -> 0
    assert agree[0, 1] == 0
    assert mm[0, 1] == 1


def test_build_fused_flood_raster_rules():
    """Verify fusion rules: classical_plus_ml, intersection, union."""
    classical = np.array([[1, 1, 0, 0, 1, 255]], dtype=np.uint8)
    ml =        np.array([[1, 0, 1, 0, 255, 255]], dtype=np.uint8)

    # 1. classical_plus_ml
    fused_cpm = build_fused_flood_raster(classical, ml, rule="classical_plus_ml")
    # col 0: both flood -> 1
    assert fused_cpm[0, 0] == 1
    # col 1: classical flood -> 1
    assert fused_cpm[0, 1] == 1
    # col 2: ml flood -> 1
    assert fused_cpm[0, 2] == 1
    # col 3: neither -> 0
    assert fused_cpm[0, 3] == 0
    # col 4: classical flood, ml unavailable -> 1
    assert fused_cpm[0, 4] == 1
    # col 5: both nodata -> 255
    assert fused_cpm[0, 5] == 255

    # 2. intersection
    fused_int = build_fused_flood_raster(classical, ml, rule="intersection")
    # col 0: both -> 1
    assert fused_int[0, 0] == 1
    # col 1: classical only -> 0
    assert fused_int[0, 1] == 0
    # col 2: ml only -> 0
    assert fused_int[0, 2] == 0
    # col 4: ml unavailable, falls back to classical -> 1
    assert fused_int[0, 4] == 1

    # 3. union
    fused_uni = build_fused_flood_raster(classical, ml, rule="union")
    assert fused_uni[0, 0] == 1
    assert fused_uni[0, 1] == 1
    assert fused_uni[0, 2] == 1
    assert fused_uni[0, 3] == 0
    assert fused_uni[0, 4] == 1


def test_run_fusion_creates_rasters_and_meta(tmp_path: Path):
    """End-to-end run_fusion creates all rasters and updates meta.json."""
    out_dir = tmp_path / "outputs_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    import grid
    ref = grid.reference_grid()
    prof = {
        "driver": "GTiff",
        "height": ref["height"],
        "width": ref["width"],
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "crs": ref["crs"],
        "transform": ref["transform"],
    }

    c_arr = np.zeros((ref["height"], ref["width"]), dtype=np.uint8)
    c_arr[5:10, 5:10] = 1
    m_arr = np.zeros((ref["height"], ref["width"]), dtype=np.uint8)
    m_arr[7:12, 7:12] = 1
    m_arr[15:20, :] = 255  # cloudy region

    with rasterio.open(out_dir / "flood_classical.tif", "w", **prof) as dst: dst.write(c_arr, 1)
    with rasterio.open(out_dir / "flood_ml.tif", "w", **prof) as dst: dst.write(m_arr, 1)

    meta = {"event": "Assam 2022"}
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    res = run_fusion(outputs_dir=out_dir, rule="classical_plus_ml", force=True)

    assert (out_dir / "agreement.tif").exists()
    assert (out_dir / "method_mask.tif").exists()
    assert (out_dir / "flood_fused.tif").exists()

    assert res["rule"] == "classical_plus_ml"
    assert res["classical_flood_pixels"] == 25
    assert "agreement_counts" in res

    meta_updated = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert "fusion" in meta_updated
    assert meta_updated["fusion"]["rule"] == "classical_plus_ml"
