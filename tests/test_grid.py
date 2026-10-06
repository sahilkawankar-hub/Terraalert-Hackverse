"""
Tests for grid.py — synthetic raster generation and reference grid alignment.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import rasterio
from affine import Affine

import config
from grid import (
    GridAlignmentError,
    align_to_grid,
    check_aligned,
    reference_grid,
)


@pytest.fixture
def ref_grid() -> dict:
    """Fixture providing reference grid parameters."""
    return reference_grid()


def test_reference_grid_properties(ref_grid):
    """Verify reference grid schema and alignment constraints."""
    assert "crs" in ref_grid
    assert "transform" in ref_grid
    assert "width" in ref_grid
    assert "height" in ref_grid

    assert ref_grid["crs"] == config.GRID_CRS
    t = ref_grid["transform"]
    assert math.isclose(t.a, config.PIXEL_SIZE, abs_tol=1e-5)
    assert math.isclose(t.e, -config.PIXEL_SIZE, abs_tol=1e-5)

    # Origin should be snapped to exact multiples of config.PIXEL_SIZE
    assert t.c % config.PIXEL_SIZE == 0
    assert t.f % config.PIXEL_SIZE == 0

    assert isinstance(ref_grid["width"], int) and ref_grid["width"] > 0
    assert isinstance(ref_grid["height"], int) and ref_grid["height"] > 0


def _create_synthetic_raster(
    path: Path,
    crs: str,
    transform: Affine,
    width: int,
    height: int,
    dtype: str = "float32",
    nodata: float = -9999.0,
    fill_value: float = 1.0,
):
    """Helper to create a synthetic single-band GeoTIFF."""
    data = np.full((height, width), fill_value=fill_value, dtype=dtype)
    profile = {
        "driver": "GTiff",
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_check_aligned_success(tmp_path, ref_grid):
    """check_aligned should pass for a raster matching reference grid exactly."""
    valid_tif = tmp_path / "valid.tif"
    _create_synthetic_raster(
        valid_tif,
        crs=ref_grid["crs"],
        transform=ref_grid["transform"],
        width=ref_grid["width"],
        height=ref_grid["height"],
    )

    # Should not raise
    check_aligned(valid_tif)


def test_check_aligned_crs_mismatch(tmp_path, ref_grid):
    """check_aligned should raise GridAlignmentError if CRS differs."""
    bad_crs_tif = tmp_path / "bad_crs.tif"
    _create_synthetic_raster(
        bad_crs_tif,
        crs="EPSG:4326",
        transform=ref_grid["transform"],
        width=ref_grid["width"],
        height=ref_grid["height"],
    )

    with pytest.raises(GridAlignmentError, match="CRS mismatch"):
        check_aligned(bad_crs_tif)


def test_check_aligned_dimension_mismatch(tmp_path, ref_grid):
    """check_aligned should raise GridAlignmentError if dimensions differ."""
    bad_dim_tif = tmp_path / "bad_dim.tif"
    _create_synthetic_raster(
        bad_dim_tif,
        crs=ref_grid["crs"],
        transform=ref_grid["transform"],
        width=ref_grid["width"] - 10,
        height=ref_grid["height"],
    )

    with pytest.raises(GridAlignmentError, match="width mismatch"):
        check_aligned(bad_dim_tif)


def test_check_aligned_origin_mismatch(tmp_path, ref_grid):
    """check_aligned should raise GridAlignmentError if origin is shifted."""
    t = ref_grid["transform"]
    shifted_transform = Affine(t.a, t.b, t.c + 5.0, t.d, t.e, t.f)
    bad_origin_tif = tmp_path / "bad_origin.tif"
    _create_synthetic_raster(
        bad_origin_tif,
        crs=ref_grid["crs"],
        transform=shifted_transform,
        width=ref_grid["width"],
        height=ref_grid["height"],
    )

    with pytest.raises(GridAlignmentError, match="x-origin mismatch"):
        check_aligned(bad_origin_tif)


def test_align_to_grid(tmp_path, ref_grid):
    """align_to_grid should reproject an arbitrary raster to the reference grid."""
    # Create an arbitrary raster in EPSG:4326 covering roughly the AOI
    west, south, east, north = config.AOI_BBOX
    src_w, src_h = 200, 200
    res_x = (east - west) / src_w
    res_y = (north - south) / src_h
    src_transform = Affine(res_x, 0, west, 0, -res_y, north)
    src_tif = tmp_path / "unaligned_src.tif"

    _create_synthetic_raster(
        src_tif,
        crs="EPSG:4326",
        transform=src_transform,
        width=src_w,
        height=src_h,
        fill_value=42.0,
    )

    dst_tif = tmp_path / "aligned_dst.tif"
    res = align_to_grid(src_tif, dst_tif, resampling="nearest")

    assert res == dst_tif
    assert dst_tif.exists()

    # Verify that check_aligned passes on dst_tif
    check_aligned(dst_tif)

    with rasterio.open(dst_tif) as ds:
        assert ds.width == ref_grid["width"]
        assert ds.height == ref_grid["height"]
        assert ds.crs == ref_grid["crs"]
        assert ds.count == 1
        arr = ds.read(1)
        # Inside the AOI, pixels should have been sampled from the source
        assert (arr == 42.0).any()
