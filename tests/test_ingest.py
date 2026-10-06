"""
Tests for Earth Engine ingestion and data contracts (no network required).
"""
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

import config
import grid
from backend.ingest.gee import (
    print_raster_stats,
    resample_population_preserving_count,
)


def test_population_preserving_resample(tmp_path: Path):
    """Verify that population resampling strictly preserves total population count."""
    ref_grid = grid.reference_grid()
    w, h = ref_grid["width"], ref_grid["height"]
    t = ref_grid["transform"]

    # 1. Create a synthetic coarse raw population raster (e.g. 5x5 pixels)
    # covering the same spatial extent with known total population
    target_total = 75000.0
    coarse_w, coarse_h = 20, 20
    # Map coarse cells across extent
    x_min, y_max = t.c, t.f
    x_max = x_min + w * config.PIXEL_SIZE
    y_min = y_max - h * config.PIXEL_SIZE
    coarse_ps_x = (x_max - x_min) / coarse_w
    coarse_ps_y = (y_max - y_min) / coarse_h
    coarse_transform = Affine(coarse_ps_x, 0, x_min, 0, -coarse_ps_y, y_max)

    # Random distribution of population
    np.random.seed(42)
    raw_data = np.random.uniform(50.0, 200.0, (coarse_h, coarse_w)).astype(np.float32)
    # Scale to exact target_total
    raw_data = (raw_data / np.sum(raw_data)) * target_total
    assert math.isclose(float(np.sum(raw_data)), target_total, rel_tol=1e-5)

    raw_path = tmp_path / "pop_raw.tif"
    with rasterio.open(
        raw_path,
        "w",
        driver="GTiff",
        height=coarse_h,
        width=coarse_w,
        count=1,
        dtype="float32",
        crs=config.GRID_CRS,
        transform=coarse_transform,
        nodata=-9999.0,
    ) as ds:
        ds.write(raw_data, 1)

    # 2. Resample preserving total count to reference grid
    out_path = tmp_path / "pop_aligned.tif"
    resample_population_preserving_count(raw_path, out_path, target_total)

    # 3. Verify aligned raster properties and total count conservation
    grid.check_aligned(out_path)
    with rasterio.open(out_path) as ds:
        aligned_data = ds.read(1)
        valid = (aligned_data != -9999.0) & ~np.isnan(aligned_data) & (aligned_data > 0)
        final_sum = float(np.sum(aligned_data[valid]))

    # Total must match target_total within 0.1% tolerance
    assert math.isclose(final_sum, target_total, rel_tol=1e-3)
    assert aligned_data.shape == (h, w)


def test_synthetic_db_range_contract(tmp_path: Path):
    """Test Sentinel-1 SAR dB contract: float32, nodata -9999, typical values in [-35, 5] dB."""
    ref_grid = grid.reference_grid()
    w, h = 50, 50
    t = ref_grid["transform"]

    # Synthetic SAR backscatter in dB
    np.random.seed(123)
    sar_data = np.random.uniform(-25.0, -5.0, (h, w)).astype(np.float32)
    # Add some nodata pixels
    sar_data[0:5, 0:5] = -9999.0

    test_file = tmp_path / "test_s1.tif"
    with rasterio.open(
        test_file,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=config.GRID_CRS,
        transform=t,
        nodata=-9999.0,
    ) as ds:
        ds.write(sar_data, 1)

    stats = print_raster_stats(test_file)
    assert stats["nodata_count"] == 25
    assert -26.0 <= stats["min"] <= -20.0
    assert -10.0 <= stats["max"] <= -4.0
    assert -20.0 <= stats["mean"] <= -10.0
