"""
Unit tests for backend/fusion/confidence.py.

Verifies:
  1. Flat terrain with strong margin -> High confidence.
  2. Steep slope (>15°) -> Low confidence with reason naming steep terrain.
  3. Zones with no flood pixels are handled without crashing.
  4. Fallback contract when inputs are missing.
  5. Schema contract of conf_detail (agreement_ratio, terrain_penalty, time_penalty, margin_score, method_penalty).
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from backend.fusion.confidence import (
    add_zone_confidence,
    compute_pixel_confidence_array,
    compute_time_penalty,
)


@pytest.fixture
def mock_outputs_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with synthetic rasters and metadata."""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 100x100 raster in metric CRS (EPSG:32646)
    transform = from_origin(200000.0, 2900000.0, 20.0, 20.0)
    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32646",
        "transform": transform,
    }

    # Flat, strong margin in north (rows 0:50)
    # Steep, low margin in south (rows 50:100)
    pre = np.full((100, 100), -12.0, dtype=np.float32)
    post = np.full((100, 100), -24.0, dtype=np.float32)  # deep water (-24 dB < -18 dB)
    post[50:100, :] = -18.5                             # shallow margin in south

    slope = np.full((100, 100), 1.0, dtype=np.float32)   # flat in north
    slope[50:100, :] = 25.0                             # steep in south (> 15°)

    prof_uint8 = profile.copy()
    prof_uint8.update(dtype="uint8", nodata=255)

    flood = np.zeros((100, 100), dtype=np.uint8)
    flood[10:40, 10:40] = 1   # flood patch in north (flat, strong margin)
    flood[60:90, 60:90] = 1   # flood patch in south (steep, weak margin)

    perm = np.zeros((100, 100), dtype=np.uint8)

    with rasterio.open(out_dir / "pre.tif", "w", **profile) as dst:
        dst.write(pre, 1)
    with rasterio.open(out_dir / "post.tif", "w", **profile) as dst:
        dst.write(post, 1)
    with rasterio.open(out_dir / "slope.tif", "w", **profile) as dst:
        dst.write(slope, 1)
    with rasterio.open(out_dir / "flood_classical.tif", "w", **prof_uint8) as dst:
        dst.write(flood, 1)
    with rasterio.open(out_dir / "perm_water.tif", "w", **prof_uint8) as dst:
        dst.write(perm, 1)

    meta = {
        "event": "Test Event",
        "time_gap_days": 12,
        "grid_crs": "EPSG:32646",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    return out_dir


def test_add_zone_confidence_flat_strong_margin(mock_outputs_dir: Path):
    """Zone over flat terrain with strong margin must score High."""
    # Zone 1: covering north flood patch (x: 200200-200800, y: 2899200-2899800)
    geom_north = box(200200.0, 2899200.0, 200800.0, 2899800.0)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["zone_flat"]},
        geometry=[geom_north],
        crs="EPSG:32646",
    )

    scored = add_zone_confidence(zones, outputs_dir=mock_outputs_dir)

    assert len(scored) == 1
    assert scored.iloc[0]["confidence"] == "High"
    assert scored.iloc[0]["confidence_score"] >= 0.70
    assert scored.iloc[0]["conf_detail"]["terrain_penalty"] == 0.0


def test_add_zone_confidence_steep_slope_reason(mock_outputs_dir: Path):
    """Zone over steep slope must score Low with a reason explicitly mentioning steep terrain."""
    # Zone 2: covering south flood patch (x: 201200-201800, y: 2898200-2898800)
    geom_south = box(201200.0, 2898200.0, 201800.0, 2898800.0)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["zone_steep"]},
        geometry=[geom_south],
        crs="EPSG:32646",
    )

    scored = add_zone_confidence(zones, outputs_dir=mock_outputs_dir)

    assert len(scored) == 1
    assert scored.iloc[0]["confidence"] == "Low"
    assert scored.iloc[0]["confidence_score"] < 0.40
    reason = scored.iloc[0]["reason"].lower()
    assert "steep terrain" in reason
    assert scored.iloc[0]["conf_detail"]["terrain_penalty"] > 0.5


def test_add_zone_confidence_no_flood_pixels(mock_outputs_dir: Path):
    """Zone with no flood pixels must be handled gracefully without crashing."""
    # Zone 3: area with no flood pixels (x: 200000-200180, y: 2899820-2900000)
    geom_dry = box(200000.0, 2899820.0, 200180.0, 2900000.0)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["zone_dry"]},
        geometry=[geom_dry],
        crs="EPSG:32646",
    )

    scored = add_zone_confidence(zones, outputs_dir=mock_outputs_dir)

    assert len(scored) == 1
    assert scored.iloc[0]["confidence"] == "Medium"
    assert scored.iloc[0]["confidence_score"] == 0.50
    assert "no flood pixels" in scored.iloc[0]["reason"].lower()


def test_add_zone_confidence_fallback_missing_inputs(tmp_path: Path):
    """When input rasters are missing, fallback contract must apply."""
    empty_dir = tmp_path / "empty_outputs"
    empty_dir.mkdir()

    geom = box(0.0, 0.0, 100.0, 100.0)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["zone_fallback"]},
        geometry=[geom],
        crs="EPSG:32646",
    )

    scored = add_zone_confidence(zones, outputs_dir=empty_dir)

    assert scored.iloc[0]["confidence"] == "Medium"
    assert scored.iloc[0]["confidence_score"] == 0.50
    assert scored.iloc[0]["reason"] == "confidence module not run yet"


def test_conf_detail_contract_fields(mock_outputs_dir: Path):
    """conf_detail dict must contain all 5 required fields."""
    geom = box(200200.0, 2899200.0, 200800.0, 2899800.0)
    zones = gpd.GeoDataFrame(
        {"zone_id": ["z1"]},
        geometry=[geom],
        crs="EPSG:32646",
    )
    scored = add_zone_confidence(zones, outputs_dir=mock_outputs_dir)
    detail = scored.iloc[0]["conf_detail"]

    required_keys = {"agreement_ratio", "terrain_penalty", "time_penalty", "margin_score", "method_penalty"}
    assert required_keys == set(detail.keys())
    assert detail["agreement_ratio"] == 1.0
    assert detail["method_penalty"] == 0.0


def test_compute_time_penalty():
    """Verify time penalty bounds."""
    pen_0, score_0 = compute_time_penalty(12)
    assert pen_0 == 0.0
    assert score_0 == 1.0

    pen_long, score_long = compute_time_penalty(100)
    assert pen_long == 0.5
    assert score_long == 0.5
