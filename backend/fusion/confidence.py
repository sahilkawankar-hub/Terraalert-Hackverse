"""
TerraAlert — Decision fusion and confidence scoring.

Produces pixel-level and zone-level confidence scores combining five signals:
  1. Multi-Method Agreement (weight in CONF_WEIGHTS['agreement'], default 0.35):
     Share of flooded pixels in zone where both SAR and Optical ML agree (agreement == 2):
       agreement_ratio = count(flood & agreement == 2) / count(flood)
     Where only single-method detection is available or methods disagree, agreement_ratio is 0.

  2. Detection Margin (weight in CONF_WEIGHTS['margin'], default 0.20):
     Distance of SAR post-event backscatter and pre/post differencing from decision boundaries:
       margin_post = clip((POST_DB_MAX - post) / 6.0, 0.0, 1.0)
       margin_diff = clip((DIFF_DB_MAX - (post - pre)) / 6.0, 0.0, 1.0)
       margin_score = 0.5 * margin_post + 0.5 * margin_diff
     Pixels far below threshold (deep water, strong change) are more certain.

  3. Terrain Stability (weight in CONF_WEIGHTS['terrain'], default 0.25):
     Radar distortion risk from slope layover/shadow:
       terrain_penalty = clip((slope - 5.0) / (STEEP_FLAG_DEG - 5.0), 0.0, 1.0)
       terrain_score = 1.0 - terrain_penalty
     Slopes above STEEP_FLAG_DEG (15°) receive maximum penalty.

  4. Temporal Proximity (weight in CONF_WEIGHTS['time'], default 0.20):
     Time gap between pre- and post-event acquisitions (nominal repeat = 12 days):
       time_penalty = clip((time_gap_days - 12.0) / (60.0 - 12.0) * 0.5, 0.0, 0.5)
       time_score = 1.0 - time_penalty

  5. Method Availability Penalty (METHOD_PENALTY_MAX, default 0.15):
     Penalty deducted if optical ML is unavailable for the zone (monsoon clouds / nodata):
       method_penalty = (share_of_zone_with_ml_unavailable) * METHOD_PENALTY_MAX

Combined Confidence Formula:
  base_score = (w_agree * agreement_ratio) + (w_margin * margin_score) + (w_terrain * terrain_score) + (w_time * time_score)
  confidence_score = clip(base_score - method_penalty, 0.0, 1.0)

Zone-Level Interface Contract (Signature MUST NOT change):
  add_zone_confidence(zones: GeoDataFrame, outputs_dir: Optional[Path] = None) -> GeoDataFrame
  Adds columns:
    - confidence ('High' | 'Medium' | 'Low')
    - confidence_score (float 0.0–1.0)
    - reason (str)
    - conf_detail (dict: agreement_ratio, terrain_penalty, time_penalty, margin_score, method_penalty)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows

import config
import grid
from backend.common.log import get_logger
from backend.common.meta import load_meta, update_meta

logger = get_logger("fusion.confidence")


def compute_time_penalty(time_gap_days: int) -> Tuple[float, float]:
    """
    Compute time penalty and time score from satellite acquisition gap.
    Returns (time_penalty, time_score).
    """
    penalty = float(np.clip((time_gap_days - 12.0) / 48.0 * 0.5, 0.0, 0.5))
    score = 1.0 - penalty
    return penalty, score


def compute_pixel_confidence_array(
    pre: np.ndarray,
    post: np.ndarray,
    slope: np.ndarray,
    valid: np.ndarray,
    time_gap_days: int = 12,
    agreement: Optional[np.ndarray] = None,
    method_mask: Optional[np.ndarray] = None,
    post_db_max: float = config.POST_DB_MAX,
    diff_db_max: float = config.DIFF_DB_MAX,
    steep_flag_deg: float = config.STEEP_FLAG_DEG,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute continuous float32 confidence map across raster grid.

    Returns
    -------
    conf_map : np.ndarray (float32)
        0.0 to 1.0 confidence score (nodata=-9999.0).
    margin_map : np.ndarray (float32)
    time_penalty : float
    terrain_penalty_map : np.ndarray (float32)
    """
    # 1. Detection Margin Score
    margin_post = np.clip((post_db_max - post) / 6.0, 0.0, 1.0)
    diff = post - pre
    margin_diff = np.clip((diff_db_max - diff) / 6.0, 0.0, 1.0)
    margin_score = 0.5 * margin_post + 0.5 * margin_diff

    # 2. Terrain Penalty (slopes above 5 deg scale to STEEP_FLAG_DEG)
    terrain_penalty = np.clip((slope - 5.0) / max(steep_flag_deg - 5.0, 1.0), 0.0, 1.0)
    terrain_score = 1.0 - terrain_penalty

    # 3. Time Score
    time_penalty, time_score = compute_time_penalty(time_gap_days)

    # Weights
    w_agree = config.CONF_WEIGHTS.get("agreement", 0.35)
    w_margin = config.CONF_WEIGHTS.get("margin", 0.20)
    w_terrain = config.CONF_WEIGHTS.get("terrain", 0.25)
    w_time = config.CONF_WEIGHTS.get("time", 0.20)

    # 4. Multi-method agreement & method penalty if rasters available
    if agreement is not None:
        agree_score = np.where(agreement == 2, 1.0, np.where(agreement == 1, 0.4, 0.0)).astype(np.float32)
    else:
        agree_score = margin_score  # fallback to detection margin certainty

    if method_mask is not None:
        pen_max = getattr(config, "METHOD_PENALTY_MAX", 0.15)
        method_pen_map = np.where(method_mask == 1, pen_max, 0.0).astype(np.float32)
    else:
        method_pen_map = np.zeros_like(margin_score)

    combined = (
        (w_agree * agree_score)
        + (w_margin * margin_score)
        + (w_terrain * terrain_score)
        + (w_time * time_score)
        - method_pen_map
    )
    combined = np.clip(combined, 0.0, 1.0).astype(np.float32)

    conf_out = np.full(post.shape, -9999.0, dtype=np.float32)
    conf_out[valid] = combined[valid]

    return conf_out, margin_score, time_penalty, terrain_penalty


def generate_confidence_raster(
    outputs_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Generate outputs/confidence_pixel.tif using inputs from outputs/.
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    if out_path is None:
        out_path = outputs_dir / "confidence_pixel.tif"

    if out_path.exists() and not force:
        logger.info("%s exists. Skipping confidence raster generation (use --force).", out_path)
        return out_path

    pre_path = outputs_dir / "pre.tif"
    post_path = outputs_dir / "post.tif"
    slope_path = outputs_dir / "slope.tif"
    perm_path = outputs_dir / "perm_water.tif"

    for p in (pre_path, post_path, slope_path, perm_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing input raster for confidence calculation: {p}")

    grid.check_aligned(pre_path, post_path, slope_path, perm_path)

    with rasterio.open(pre_path) as src:
        pre = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(post_path) as src:
        post = src.read(1)
    with rasterio.open(slope_path) as src:
        slope = src.read(1)
    with rasterio.open(perm_path) as src:
        perm = src.read(1)

    agreement_path = outputs_dir / "agreement.tif"
    method_mask_path = outputs_dir / "method_mask.tif"
    agreement = None
    method_mask = None
    if agreement_path.exists():
        with rasterio.open(agreement_path) as src:
            agreement = src.read(1)
    if method_mask_path.exists():
        with rasterio.open(method_mask_path) as src:
            method_mask = src.read(1)

    meta_path = outputs_dir / "meta.json"
    meta = load_meta(meta_path)
    time_gap_days = int(meta.get("time_gap_days", 12))

    valid = (
        (pre != -9999.0) & (~np.isnan(pre)) &
        (post != -9999.0) & (~np.isnan(post)) &
        (slope != -9999.0) & (~np.isnan(slope))
    )

    conf_arr, _, _, _ = compute_pixel_confidence_array(
        pre=pre,
        post=post,
        slope=slope,
        valid=valid,
        time_gap_days=time_gap_days,
        agreement=agreement,
        method_mask=method_mask,
        post_db_max=config.POST_DB_MAX,
        diff_db_max=config.DIFF_DB_MAX,
        steep_flag_deg=config.STEEP_FLAG_DEG,
    )

    profile.update(
        dtype=rasterio.float32,
        nodata=-9999.0,
        count=1,
        compress="lzw",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(conf_arr, 1)

    logger.info("Saved pixel confidence raster to %s", out_path)
    return out_path


def _generate_reason(
    label: str,
    terrain_penalty: float,
    time_penalty: float,
    margin_score: float,
    time_gap_days: int,
    agreement_ratio: float = 1.0,
    method_penalty: float = 0.0,
) -> str:
    """Construct human-readable reason string highlighting factors costing the most."""
    w_agree = config.CONF_WEIGHTS.get("agreement", 0.35)
    w_margin = config.CONF_WEIGHTS.get("margin", 0.20)
    w_terrain = config.CONF_WEIGHTS.get("terrain", 0.25)
    w_time = config.CONF_WEIGHTS.get("time", 0.20)

    cost_agree = w_agree * (1.0 - agreement_ratio)
    cost_terrain = w_terrain * terrain_penalty
    cost_margin = w_margin * (1.0 - margin_score)
    cost_time = w_time * time_penalty
    cost_method = method_penalty

    costs = [
        ("terrain", cost_terrain),
        ("agreement", cost_agree),
        ("method", cost_method),
        ("margin", cost_margin),
        ("time", cost_time),
    ]
    costs.sort(key=lambda x: x[1], reverse=True)

    reasons: list[str] = []
    for factor, cost in costs[:2]:
        if cost > 0.04:
            if factor == "agreement" and agreement_ratio < 0.5:
                reasons.append("single-method detection (methods disagree)")
            elif factor == "method" and method_penalty > 0.04:
                reasons.append("optical ML unavailable for zone")
            elif factor == "terrain" and terrain_penalty > 0.15:
                pct = int(round(terrain_penalty * 100))
                reasons.append(f"{pct}% of flooded area on steep terrain (>15°)")
            elif factor == "margin" and margin_score < 0.65:
                reasons.append("flood pixels close to SAR detection threshold")
            elif factor == "time" and time_penalty > 0.15:
                reasons.append(f"{time_gap_days}-day temporal gap between passes")

    if not reasons:
        if agreement_ratio >= 0.8 and method_penalty == 0.0:
            return f"{label}: Both SAR and optical ML confirm flood on flat terrain with high margin"
        return f"{label}: Strong radar backscatter drop on flat terrain with high margin"

    joined = "; ".join(reasons)
    return f"{label}: {joined}"


def add_zone_confidence(
    zones: gpd.GeoDataFrame,
    outputs_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Compute and assign confidence metrics to a GeoDataFrame of zones.

    Interface contract:
      Input: zones in config.GRID_CRS.
      Outputs: GeoDataFrame with columns:
        - confidence ('High' | 'Medium' | 'Low')
        - confidence_score (float 0.0–1.0)
        - reason (str)
        - conf_detail (dict: agreement_ratio, terrain_penalty, time_penalty, margin_score, method_penalty)
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR

    # Resolve flood raster: flood_fused.tif if exists, else flood_classical.tif
    flood_path = outputs_dir / "flood_fused.tif"
    if not flood_path.exists():
        flood_path = outputs_dir / "flood_classical.tif"

    pre_path = outputs_dir / "pre.tif"
    post_path = outputs_dir / "post.tif"
    slope_path = outputs_dir / "slope.tif"
    meta_path = outputs_dir / "meta.json"

    # Fallback check per AGENT_CONTEXT: if modules/inputs do not exist
    if not flood_path.exists() or not pre_path.exists() or not post_path.exists() or not slope_path.exists():
        logger.warning("Required rasters missing for confidence calculation. Applying fallback contract.")
        zones = zones.copy()
        zones["confidence"] = "Medium"
        zones["confidence_score"] = 0.50
        zones["reason"] = "confidence module not run yet"
        zones["conf_detail"] = [
            {
                "agreement_ratio": 1.0,
                "terrain_penalty": 0.0,
                "time_penalty": 0.0,
                "margin_score": 0.5,
                "method_penalty": 0.0,
            }
            for _ in range(len(zones))
        ]
        if meta_path.exists():
            update_meta(meta_path, {"fallbacks.confidence": "inputs missing; assigned default 0.5"})
        return zones

    meta = load_meta(meta_path)
    time_gap_days = int(meta.get("time_gap_days", 12))
    time_pen, time_score = compute_time_penalty(time_gap_days)

    with rasterio.open(flood_path) as src_flood:
        flood_arr = src_flood.read(1)
        src_transform = src_flood.transform
        src_crs = src_flood.crs

    with rasterio.open(pre_path) as src:
        pre_arr = src.read(1)
    with rasterio.open(post_path) as src:
        post_arr = src.read(1)
    with rasterio.open(slope_path) as src:
        slope_arr = src.read(1)

    agreement_path = outputs_dir / "agreement.tif"
    method_mask_path = outputs_dir / "method_mask.tif"
    ml_path = outputs_dir / "flood_ml.tif"

    agreement_arr: Optional[np.ndarray] = None
    method_mask_arr: Optional[np.ndarray] = None

    if agreement_path.exists():
        with rasterio.open(agreement_path) as src:
            agreement_arr = src.read(1)
    if method_mask_path.exists():
        with rasterio.open(method_mask_path) as src:
            method_mask_arr = src.read(1)
    elif ml_path.exists() and agreement_arr is None:
        with rasterio.open(ml_path) as src_ml:
            ml_arr = src_ml.read(1)
            from backend.fusion.fuse import build_agreement_raster
            agreement_arr, method_mask_arr = build_agreement_raster(flood_arr, ml_arr)

    # Margin and terrain calculations
    margin_post = np.clip((config.POST_DB_MAX - post_arr) / 6.0, 0.0, 1.0)
    diff = post_arr - pre_arr
    margin_diff = np.clip((config.DIFF_DB_MAX - diff) / 6.0, 0.0, 1.0)
    margin_arr = (0.5 * margin_post + 0.5 * margin_diff).astype(np.float32)

    # Steep terrain indicator (> 15 deg)
    steep_mask = (slope_arr > config.STEEP_FLAG_DEG)

    zones_out = zones.copy()
    if zones_out.crs is None or zones_out.crs.to_string() != src_crs.to_string():
        zones_out = zones_out.to_crs(src_crs)

    conf_labels: list[str] = []
    conf_scores: list[float] = []
    reasons: list[str] = []
    conf_details: list[dict[str, Any]] = []

    for _, row in zones_out.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            conf_labels.append("Medium")
            conf_scores.append(0.50)
            reasons.append("Empty zone geometry")
            conf_details.append({
                "agreement_ratio": 1.0,
                "terrain_penalty": 0.0,
                "time_penalty": time_pen,
                "margin_score": 1.0,
                "method_penalty": 0.0,
            })
            continue

        minx, miny, maxx, maxy = geom.bounds
        win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src_transform)
        win = win.round_offsets().round_lengths()

        col_off = max(0, int(win.col_off))
        row_off = max(0, int(win.row_off))
        w = max(1, min(int(win.width), flood_arr.shape[1] - col_off))
        h = max(1, min(int(win.height), flood_arr.shape[0] - row_off))

        sub_transform = rasterio.windows.transform(
            rasterio.windows.Window(col_off, row_off, w, h),
            src_transform,
        )

        sub_geom_mask = rasterio.features.geometry_mask(
            [geom],
            out_shape=(h, w),
            transform=sub_transform,
            invert=True,
        )

        sub_flood = flood_arr[row_off:row_off + h, col_off:col_off + w]
        flood_in_zone = sub_geom_mask & (sub_flood == 1)
        n_flood = int(flood_in_zone.sum())

        if n_flood == 0:
            # Zone has no flood pixels; handle without crashing
            conf_labels.append("Medium")
            conf_scores.append(0.50)
            reasons.append("Medium: No flood pixels detected in zone")
            conf_details.append({
                "agreement_ratio": 1.0,
                "terrain_penalty": 0.0,
                "time_penalty": time_pen,
                "margin_score": 1.0,
                "method_penalty": 0.0,
            })
            continue

        sub_margin = margin_arr[row_off:row_off + h, col_off:col_off + w]
        sub_steep = steep_mask[row_off:row_off + h, col_off:col_off + w]

        mean_margin = float(sub_margin[flood_in_zone].mean())
        terrain_penalty_frac = float(sub_steep[flood_in_zone].mean())
        terrain_score = 1.0 - terrain_penalty_frac

        # 1. Multi-method agreement ratio
        if agreement_arr is not None:
            sub_agree = agreement_arr[row_off:row_off + h, col_off:col_off + w]
            both_agree_px = int(((sub_agree == 2) & flood_in_zone).sum())
            agreement_ratio = float(both_agree_px / n_flood)
        else:
            agreement_ratio = 1.0

        # 2. Method availability penalty
        if method_mask_arr is not None:
            sub_mm = method_mask_arr[row_off:row_off + h, col_off:col_off + w]
            unavail_px = int((sub_geom_mask & (sub_mm == 1)).sum())
            zone_px = int(sub_geom_mask.sum())
            unavail_frac = float(unavail_px / zone_px) if zone_px > 0 else 0.0
            pen_max = getattr(config, "METHOD_PENALTY_MAX", 0.15)
            method_pen = float(unavail_frac * pen_max)
        else:
            method_pen = 0.0

        w_agree = config.CONF_WEIGHTS.get("agreement", 0.35)
        w_margin = config.CONF_WEIGHTS.get("margin", 0.20)
        w_terrain = config.CONF_WEIGHTS.get("terrain", 0.25)
        w_time = config.CONF_WEIGHTS.get("time", 0.20)

        if agreement_arr is not None:
            base_score = (
                (w_agree * agreement_ratio)
                + (w_margin * mean_margin)
                + (w_terrain * terrain_score)
                + (w_time * time_score)
            )
        else:
            # Single-method fallback when multi-method agreement raster is absent
            base_score = (
                ((w_agree + w_margin) * mean_margin)
                + (w_terrain * terrain_score)
                + (w_time * time_score)
            )

        final_score = base_score - method_pen
        score = float(np.clip(final_score, 0.0, 1.0))

        if score >= config.CONF_HIGH:
            label = "High"
        elif score < config.CONF_LOW:
            label = "Low"
        else:
            label = "Medium"

        reason_str = _generate_reason(
            label=label,
            terrain_penalty=terrain_penalty_frac,
            time_penalty=time_pen,
            margin_score=mean_margin,
            time_gap_days=time_gap_days,
            agreement_ratio=agreement_ratio,
            method_penalty=method_pen,
        )

        detail = {
            "agreement_ratio": round(agreement_ratio, 3),
            "terrain_penalty": round(terrain_penalty_frac, 3),
            "time_penalty": round(time_pen, 3),
            "margin_score": round(mean_margin, 3),
            "method_penalty": round(method_pen, 3),
        }

        conf_labels.append(label)
        conf_scores.append(round(score, 3))
        reasons.append(reason_str)
        conf_details.append(detail)

    zones_out["confidence"] = conf_labels
    zones_out["confidence_score"] = conf_scores
    zones_out["reason"] = reasons
    zones_out["conf_detail"] = conf_details

    return zones_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confidence raster and score zones")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    # Step 1: Generate pixel confidence raster
    generate_confidence_raster(force=args.force)

    # Step 2: Test on existing zones (demo or real)
    demo_zones_path = config.OUTPUTS_DIR / "demo" / "zones.geojson"
    if demo_zones_path.exists():
        logger.info("Scoring demo zones from %s...", demo_zones_path)
        zones_gdf = gpd.read_file(demo_zones_path)
        scored_gdf = add_zone_confidence(zones_gdf)
        print("\n=== Demo Zones Confidence Summary ===")
        print(scored_gdf["confidence"].value_counts().to_string())
        print("\nSample Reasons:")
        for r in scored_gdf["reason"].head(5):
            print(f" - {r}")


if __name__ == "__main__":
    main()
