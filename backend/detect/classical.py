"""
TerraAlert — Classical SAR change detection pipeline.

Applies multi-condition thresholding to detect new inundation:
  flood = (post < POST_DB_MAX)
          AND ((post - pre) < DIFF_DB_MAX)
          AND (perm_water == 0)
          AND (slope < SLOPE_MAX_DEG)

Treats nodata (-9999 or 255) as not flood, applies morphological cleaning
(removes objects < MIN_OBJECT_PIXELS and fills holes < MIN_OBJECT_PIXELS),
supports optional Otsu thresholding on the post-event SAR backscatter,
updates metadata atomically via update_meta(), and saves outputs/flood_classical.tif.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import scipy.ndimage

import config
import grid
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("detect.classical")


def compute_otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """
    Compute Otsu threshold on continuous SAR backscatter (dB).
    Finds the threshold maximizing between-class variance.
    """
    valid_vals = values[np.isfinite(values)]
    if len(valid_vals) == 0:
        return config.POST_DB_MAX

    hist, bin_edges = np.histogram(valid_vals, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0:
        return config.POST_DB_MAX

    current_max = 0.0
    threshold = bin_centers[0]
    weight_bg = 0.0
    sum_bg = 0.0
    sum_total = float(np.dot(hist, bin_centers))

    for i in range(nbins):
        weight_bg += float(hist[i])
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += float(hist[i]) * float(bin_centers[i])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = float(bin_centers[i])

    # Bound within plausible SAR water backscatter range
    bounded_threshold = float(np.clip(threshold, -26.0, -14.0))
    logger.info(
        "Computed Otsu threshold: %.2f dB (bounded to %.2f dB)",
        threshold, bounded_threshold,
    )
    return bounded_threshold


def clean_morphology(
    mask: np.ndarray,
    min_pixels: int,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Morphologically clean binary detection mask:
      1. Remove connected components smaller than min_pixels.
      2. Fill enclosed dry holes inside flood areas smaller than min_pixels.
    """
    # 1. Remove small connected objects
    labeled, num_features = scipy.ndimage.label(mask)
    if num_features > 0:
        sizes = scipy.ndimage.sum(mask, labeled, range(num_features + 1))
        keep = sizes >= min_pixels
        cleaned = keep[labeled]
    else:
        cleaned = mask.copy()

    # 2. Fill small holes inside flood areas
    inverted = (~cleaned) & valid
    labeled_inv, num_inv = scipy.ndimage.label(inverted)
    if num_inv > 0:
        sizes_inv = scipy.ndimage.sum(inverted, labeled_inv, range(num_inv + 1))
        # Small holes are background regions with 0 < size < min_pixels
        is_small_hole = (sizes_inv > 0) & (sizes_inv < min_pixels)
        small_holes = is_small_hole[labeled_inv]
        cleaned = cleaned | small_holes

    return cleaned & valid


def detect_classical_arrays(
    pre: np.ndarray,
    post: np.ndarray,
    perm_water: np.ndarray,
    slope: np.ndarray,
    post_db_max: float = config.POST_DB_MAX,
    diff_db_max: float = config.DIFF_DB_MAX,
    slope_max_deg: float = config.SLOPE_MAX_DEG,
    min_object_pixels: int = config.MIN_OBJECT_PIXELS,
    use_otsu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run classical multi-temporal SAR flood detection on NumPy arrays.

    Returns
    -------
    flood_mask : np.ndarray (uint8)
        1 = new flood, 0 = non-flood, 255 = nodata.
    valid_mask : np.ndarray (bool)
        Valid pixel mask.
    actual_post_threshold : float
        The threshold used for post backscatter.
    """
    # Construct valid data mask (pre, post, slope validity only)
    valid = (
        (pre != -9999.0) & (~np.isnan(pre)) &
        (post != -9999.0) & (~np.isnan(post)) &
        (slope != -9999.0) & (~np.isnan(slope))
    )

    actual_post_thresh = post_db_max
    if use_otsu:
        valid_post = post[valid]
        actual_post_thresh = compute_otsu_threshold(valid_post)

    diff = post - pre

    # Primary detection rules:
    # 1. Dark post backscatter (water reflection)
    # 2. Significant backscatter drop from pre to post
    # 3. Not permanent water (perm_water == 1 is an exclusion, never a validity condition)
    # 4. Low terrain slope (< SLOPE_MAX_DEG)
    raw_flood = (
        (post < actual_post_thresh) &
        (diff < diff_db_max) &
        (perm_water != 1) &
        (slope < slope_max_deg) &
        valid
    )

    # Morphological cleaning
    cleaned_flood = clean_morphology(raw_flood, min_object_pixels, valid)

    out = np.full(post.shape, 255, dtype=np.uint8)
    out[valid & ~cleaned_flood] = 0
    out[valid & cleaned_flood] = 1

    return out, valid, actual_post_thresh


def run_classical(
    outputs_dir: Optional[Path] = None,
    use_otsu: Optional[bool] = None,
    force: bool = False,
) -> Path:
    """
    Run classical detection using rasters from outputs/ and write flood_classical.tif.
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    out_path = outputs_dir / "flood_classical.tif"

    if out_path.exists() and not force:
        logger.info("%s exists. Skipping classical detection (use --force to overwrite).", out_path)
        return out_path

    pre_path = outputs_dir / "pre.tif"
    post_path = outputs_dir / "post.tif"
    perm_path = outputs_dir / "perm_water.tif"
    slope_path = outputs_dir / "slope.tif"

    for p in (pre_path, post_path, perm_path, slope_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input raster missing: {p}")

    # Step 1: Ensure all rasters are strictly aligned to reference grid
    grid.check_aligned(pre_path, post_path, perm_path, slope_path)

    # Read rasters
    with rasterio.open(pre_path) as src:
        pre_arr = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(post_path) as src:
        post_arr = src.read(1)
    with rasterio.open(perm_path) as src:
        perm_arr = src.read(1)
    with rasterio.open(slope_path) as src:
        slope_arr = src.read(1)

    effective_use_otsu = config.USE_OTSU if use_otsu is None else use_otsu
    logger.info("Executing classical SAR change detection (use_otsu=%s)...", effective_use_otsu)

    flood_arr, valid_mask, actual_thresh = detect_classical_arrays(
        pre=pre_arr,
        post=post_arr,
        perm_water=perm_arr,
        slope=slope_arr,
        post_db_max=config.POST_DB_MAX,
        diff_db_max=config.DIFF_DB_MAX,
        slope_max_deg=config.SLOPE_MAX_DEG,
        min_object_pixels=config.MIN_OBJECT_PIXELS,
        use_otsu=effective_use_otsu,
    )

    # Write output GeoTIFF
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

    # Update metadata atomically
    meta_path = outputs_dir / "meta.json"
    update_meta(
        meta_path,
        {
            "thresholds": {
                "post_db_max": actual_thresh,
                "diff_db_max": config.DIFF_DB_MAX,
                "slope_max_deg": config.SLOPE_MAX_DEG,
                "min_object_pixels": config.MIN_OBJECT_PIXELS,
                "min_flood_pct": config.MIN_FLOOD_PCT,
                "used_otsu": effective_use_otsu,
            },
            "flood_classical_summary": {
                "flooded_pixels": n_flood,
                "flooded_pct": round(flood_pct, 2),
                "flooded_km2": round(flood_km2, 2),
            },
        },
    )

    logger.info(
        "Classical flood detection written to %s: %d pixels (%.2f%%, %.2f km²)",
        out_path, n_flood, flood_pct, flood_km2,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical SAR flood detection")
    parser.add_argument("--otsu", action="store_true", help="Enable Otsu thresholding for post backscatter")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    run_classical(use_otsu=args.otsu, force=args.force)


if __name__ == "__main__":
    main()
