"""
TerraAlert — Naive single-image baseline detection.

Classifies all pixels with post-event Sentinel-1 backscatter below POST_DB_MAX
as flood without differencing or permanent water masking:
    flood_naive = (post < POST_DB_MAX)

WARNING: This module is strictly a diagnostic baseline for comparison against
classical change detection in outputs/trap_comparison.png. It must NEVER be fed
into subsequent priority or zoning stages.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

import config
import grid
from backend.common.log import get_logger

logger = get_logger("detect.naive")


def detect_naive_array(
    post: np.ndarray,
    post_db_max: float = config.POST_DB_MAX,
    nodata_val: float = -9999.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute single-image naive flood classification.

    Parameters
    ----------
    post : np.ndarray
        Sentinel-1 post-event VV backscatter in dB.
    post_db_max : float
        Threshold below which pixels are considered dark/water.
    nodata_val : float
        Value representing invalid or unobserved data.

    Returns
    -------
    flood_mask : np.ndarray (uint8)
        1 = water, 0 = non-water, 255 = nodata.
    valid_mask : np.ndarray (bool)
        True for valid pixels.
    """
    valid = (post != nodata_val) & (~np.isnan(post))
    flood_raw = (post < post_db_max) & valid

    out = np.full(post.shape, 255, dtype=np.uint8)
    out[valid & ~flood_raw] = 0
    out[valid & flood_raw] = 1
    return out, valid


def run_naive(
    post_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    post_db_max: float = config.POST_DB_MAX,
    force: bool = False,
) -> Path:
    """
    Run naive single-image detection and write outputs/flood_naive.tif.
    """
    if post_path is None:
        post_path = config.OUTPUTS_DIR / "post.tif"
    if out_path is None:
        out_path = config.OUTPUTS_DIR / "flood_naive.tif"

    if out_path.exists() and not force:
        logger.info("%s exists. Skipping naive detection (use --force to overwrite).", out_path)
        return out_path

    if not post_path.exists():
        raise FileNotFoundError(f"Input file not found: {post_path}")

    # Verify grid alignment
    grid.check_aligned(post_path)

    with rasterio.open(post_path) as src:
        post_arr = src.read(1)
        profile = src.profile.copy()
        nodata_val = src.nodata if src.nodata is not None else -9999.0

    logger.info("Running naive detection with post_db_max=%.1f dB...", post_db_max)
    flood_arr, valid_mask = detect_naive_array(post_arr, post_db_max=post_db_max, nodata_val=nodata_val)

    profile.update(
        dtype=rasterio.uint8,
        nodata=255,
        count=1,
        compress="lzw",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(flood_arr, 1)

    n_flood = int((flood_arr == 1).sum())
    n_valid = int(valid_mask.sum())
    flood_pct = (n_flood / n_valid * 100.0) if n_valid > 0 else 0.0
    pixel_area_m2 = config.PIXEL_SIZE ** 2
    flood_km2 = (n_flood * pixel_area_m2) / 1e6

    logger.info(
        "Naive baseline saved to %s: %d pixels (%.2f%%, %.2f km²)",
        out_path, n_flood, flood_pct, flood_km2,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive single-image flood baseline")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()
    run_naive(force=args.force)


if __name__ == "__main__":
    main()
