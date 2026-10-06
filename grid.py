"""
TerraAlert — reference grid utilities.

Every raster in the pipeline must share the same CRS, pixel size, origin, and
dimensions.  This module defines that grid from config.AOI_BBOX and provides:

  reference_grid()         → dict with crs, transform, width, height
  align_to_grid(...)       → reprojects a raster to the reference grid
  check_aligned(*paths)    → raises GridAlignmentError if any file is off-grid
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

def reference_grid() -> dict[str, Any]:
    """Build the canonical raster grid from config settings.

    Returns a dict with keys:
        crs        – str, e.g. "EPSG:32646"
        transform  – Affine (top-left origin, snapped to PIXEL_SIZE multiples)
        width      – int, number of columns
        height     – int, number of rows
    """
    west, south, east, north = config.AOI_BBOX
    ps = config.PIXEL_SIZE
    crs = config.GRID_CRS

    # Reproject corner points from EPSG:4326 → target CRS
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_min, y_min = tx.transform(west, south)
    x_max, y_max = tx.transform(east, north)

    # Snap origin DOWN to nearest multiple of pixel size
    x_origin = math.floor(x_min / ps) * ps
    y_origin_top = math.ceil(y_max / ps) * ps  # top edge snaps UP

    # Compute dimensions
    width = math.ceil((x_max - x_origin) / ps)
    height = math.ceil((y_origin_top - y_min) / ps)

    transform = Affine(ps, 0, x_origin, 0, -ps, y_origin_top)

    return {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

class GridAlignmentError(Exception):
    """Raised when a raster does not match the reference grid."""


def align_to_grid(
    src_path: str | Path,
    dst_path: str | Path,
    resampling: str = "bilinear",
    dtype: str = "float32",
    nodata: float = -9999,
) -> Path:
    """Reproject *src_path* onto the reference grid and write to *dst_path*.

    Parameters
    ----------
    src_path : path to input raster
    dst_path : path to output raster (will be overwritten)
    resampling : "bilinear" for continuous data, "nearest" for masks
    dtype : numpy dtype string for the output
    nodata : nodata value to fill unmapped pixels

    Returns
    -------
    Path to the written file (same as *dst_path*).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    grid = reference_grid()
    resamp = Resampling[resampling]

    with rasterio.open(src_path) as src:
        band_count = src.count

        dst_profile = {
            "driver": "GTiff",
            "crs": grid["crs"],
            "transform": grid["transform"],
            "width": grid["width"],
            "height": grid["height"],
            "count": band_count,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
        }

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for band_idx in range(1, band_count + 1):
                src_data = src.read(band_idx)
                dst_data = np.full(
                    (grid["height"], grid["width"]),
                    fill_value=nodata,
                    dtype=dtype,
                )
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    src_nodata=src.nodata,
                    dst_nodata=nodata,
                    resampling=resamp,
                )
                dst.write(dst_data, band_idx)

    logger.info("Aligned %s → %s (%s, %s)", src_path.name, dst_path.name, dtype, resampling)
    return dst_path


def check_aligned(*paths: str | Path) -> None:
    """Verify that every given raster matches the reference grid exactly.

    Checks CRS, pixel-size (transform a and e), origin (transform c and f),
    width, and height.  Raises :class:`GridAlignmentError` on the first
    mismatch.
    """
    grid = reference_grid()
    ref_crs = grid["crs"]
    ref_t = grid["transform"]
    ref_w = grid["width"]
    ref_h = grid["height"]

    for p in paths:
        p = Path(p)
        with rasterio.open(p) as ds:
            # CRS
            if str(ds.crs) != ref_crs:
                raise GridAlignmentError(
                    f"{p.name}: CRS mismatch — got {ds.crs}, expected {ref_crs}"
                )
            # Transform components (pixel size and origin)
            t = ds.transform
            if not math.isclose(t.a, ref_t.a, rel_tol=1e-6):
                raise GridAlignmentError(
                    f"{p.name}: pixel width mismatch — got {t.a}, expected {ref_t.a}"
                )
            if not math.isclose(t.e, ref_t.e, rel_tol=1e-6):
                raise GridAlignmentError(
                    f"{p.name}: pixel height mismatch — got {t.e}, expected {ref_t.e}"
                )
            if not math.isclose(t.c, ref_t.c, abs_tol=0.01):
                raise GridAlignmentError(
                    f"{p.name}: x-origin mismatch — got {t.c}, expected {ref_t.c}"
                )
            if not math.isclose(t.f, ref_t.f, abs_tol=0.01):
                raise GridAlignmentError(
                    f"{p.name}: y-origin mismatch — got {t.f}, expected {ref_t.f}"
                )
            # Dimensions
            if ds.width != ref_w:
                raise GridAlignmentError(
                    f"{p.name}: width mismatch — got {ds.width}, expected {ref_w}"
                )
            if ds.height != ref_h:
                raise GridAlignmentError(
                    f"{p.name}: height mismatch — got {ds.height}, expected {ref_h}"
                )

    logger.info("check_aligned passed for %d file(s)", len(paths))


# ---------------------------------------------------------------------------
# CLI: python grid.py  → prints grid info
# ---------------------------------------------------------------------------

def _main() -> None:
    """Print the reference grid parameters."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    g = reference_grid()
    print("=" * 60)
    print("TerraAlert Reference Grid")
    print("=" * 60)
    print(f"  CRS         : {g['crs']}")
    print(f"  Pixel size  : {config.PIXEL_SIZE} m")
    print(f"  Width       : {g['width']} px")
    print(f"  Height      : {g['height']} px")
    print(f"  Transform   : {g['transform']}")
    t = g["transform"]
    print(f"  Origin (x,y): ({t.c:.2f}, {t.f:.2f})")
    extent_x = g["width"] * config.PIXEL_SIZE / 1000
    extent_y = g["height"] * config.PIXEL_SIZE / 1000
    print(f"  Extent      : {extent_x:.1f} km × {extent_y:.1f} km")
    print(f"  Total pixels: {g['width'] * g['height']:,}")
    print("=" * 60)


if __name__ == "__main__":
    _main()
