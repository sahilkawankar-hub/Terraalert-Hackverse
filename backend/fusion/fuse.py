"""
TerraAlert — Decision fusion of Classical SAR and Optical ML detections.

Combines SAR classical thresholding (outputs/flood_classical.tif) and Sentinel-2
Prithvi EO foundation model predictions (outputs/flood_ml.tif).

Outputs produced:
  outputs/agreement.tif:
    0   = neither method flags flood
    1   = exactly one method flags flood
    2   = both methods flag flood
    255 = nodata (neither method is valid)
    Where ML is 255 but classical is valid, it is treated as single-method detection.

  outputs/method_mask.tif:
    0   = ML available (both methods evaluated)
    1   = ML unavailable (single-method classical SAR only)
    255 = nodata in both

  outputs/flood_fused.tif:
    Final fused uint8 flood mask based on config.FUSION_RULE:
      - "classical_plus_ml": classical SAR plus ML-only pixels (with probability >= ML_PROB_HIGH
        or flood_ml == 1); classical only where ML is unavailable.
      - "intersection": both methods flag flood where ML available; classical only where ML unavailable.
      - "union": either method flags flood; classical only where ML unavailable.
    Nodata = 255.

All rasters are strictly verified to conform to grid.reference_grid().
Rule and fusion statistics are written to outputs/meta.json under "fusion".
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio

import config
import grid
from backend.common.log import get_logger
from backend.common.meta import load_meta, update_meta

logger = get_logger("fusion.fuse")


def build_agreement_raster(
    classical: np.ndarray,
    ml: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build multi-method agreement array and method availability mask.

    Parameters
    ----------
    classical : np.ndarray
        uint8 array (1 = flood, 0 = non-flood, 255 = nodata).
    ml : np.ndarray
        uint8 array (1 = flood, 0 = non-flood, 255 = unusable/cloud/nodata).

    Returns
    -------
    agreement : np.ndarray
        uint8 array:
          0 = neither method flags flood
          1 = exactly one method flags flood
          2 = both methods flag flood
          255 = nodata / unusable
    method_mask : np.ndarray
        uint8 array:
          0 = ML available
          1 = ML unavailable
          255 = both nodata
    """
    agreement = np.full(classical.shape, 255, dtype=np.uint8)
    method_mask = np.full(classical.shape, 255, dtype=np.uint8)

    # 1. Both methods valid
    both_valid = (classical != 255) & (ml != 255)
    c_flood = (classical == 1)
    m_flood = (ml == 1)

    # Both flag flood
    agreement[both_valid & c_flood & m_flood] = 2
    # Exactly one flags flood
    agreement[both_valid & (c_flood ^ m_flood)] = 1
    # Neither flags flood
    agreement[both_valid & (~c_flood) & (~m_flood)] = 0
    method_mask[both_valid] = 0

    # 2. ML unavailable (ml == 255) but classical is valid (classical != 255)
    # Per contract: treat as single-method detection
    ml_unavail_c_valid = (classical != 255) & (ml == 255)
    agreement[ml_unavail_c_valid & c_flood] = 1
    agreement[ml_unavail_c_valid & (~c_flood)] = 0
    method_mask[ml_unavail_c_valid] = 1

    # 3. Classical invalid (classical == 255) but ML is valid (ml != 255)
    c_unavail_ml_valid = (classical == 255) & (ml != 255)
    agreement[c_unavail_ml_valid & m_flood] = 1
    agreement[c_unavail_ml_valid & (~m_flood)] = 0
    method_mask[c_unavail_ml_valid] = 0

    return agreement, method_mask


def build_fused_flood_raster(
    classical: np.ndarray,
    ml: np.ndarray,
    rule: str = config.FUSION_RULE,
    prob_ml: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build fused flood mask according to the specified fusion rule.

    Parameters
    ----------
    classical : np.ndarray
        uint8 array (1 = flood, 0 = non-flood, 255 = nodata).
    ml : np.ndarray
        uint8 array (1 = flood, 0 = non-flood, 255 = nodata).
    rule : str
        "classical_plus_ml", "intersection", or "union".
    prob_ml : Optional[np.ndarray]
        float32 array of ML flood probabilities (0.0 to 1.0).

    Returns
    -------
    fused : np.ndarray
        uint8 array (1 = flood, 0 = non-flood, 255 = nodata).
    """
    fused = np.full(classical.shape, 255, dtype=np.uint8)

    c_valid = (classical != 255)
    m_valid = (ml != 255)
    any_valid = c_valid | m_valid
    fused[any_valid] = 0

    c_flood = (classical == 1)
    m_flood = (ml == 1)

    if prob_ml is not None:
        m_flood_high = m_flood & (prob_ml >= config.ML_PROB_HIGH)
    else:
        m_flood_high = m_flood

    if rule == "classical_plus_ml":
        # Classical SAR flood + ML-only flood where ML is available;
        # Classical only where ML is unavailable.
        is_flood = (
            (c_flood) |
            (m_valid & m_flood_high)
        )
        fused[any_valid & is_flood] = 1

    elif rule == "intersection":
        # Where both available: both must agree.
        # Where ML unavailable: fall back to classical only.
        is_flood = (
            (c_valid & m_valid & c_flood & m_flood) |
            (c_valid & (~m_valid) & c_flood)
        )
        fused[any_valid & is_flood] = 1

    elif rule == "union":
        # Either method flags flood
        is_flood = (
            (c_valid & c_flood) |
            (m_valid & m_flood)
        )
        fused[any_valid & is_flood] = 1

    else:
        raise ValueError(f"Unknown fusion rule: '{rule}'. Must be 'classical_plus_ml', 'intersection', or 'union'.")

    return fused


def run_fusion(
    outputs_dir: Optional[Path] = None,
    rule: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Execute fusion pipeline: produces agreement.tif, method_mask.tif, and flood_fused.tif.
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    if rule is None:
        rule = config.FUSION_RULE

    out_agreement = outputs_dir / "agreement.tif"
    out_method_mask = outputs_dir / "method_mask.tif"
    out_fused = outputs_dir / "flood_fused.tif"

    if out_fused.exists() and out_agreement.exists() and out_method_mask.exists() and not force:
        logger.info("Fused outputs already exist. Skipping (use --force to overwrite).")
        meta = load_meta(outputs_dir / "meta.json")
        return meta.get("fusion", {})

    path_classical = outputs_dir / "flood_classical.tif"
    path_ml = outputs_dir / "flood_ml.tif"

    if not path_classical.exists():
        raise FileNotFoundError(f"Missing classical flood mask: {path_classical}")
    if not path_ml.exists():
        raise FileNotFoundError(f"Missing ML flood mask: {path_ml}")

    # Check grid alignment
    grid.check_aligned(path_classical, path_ml)

    with rasterio.open(path_classical) as src_c:
        classical = src_c.read(1)
        profile = src_c.profile.copy()

    with rasterio.open(path_ml) as src_m:
        ml = src_m.read(1)

    # Optional ML probability map
    prob_path = outputs_dir / "prob_ml.tif"
    prob_ml = None
    if prob_path.exists():
        with rasterio.open(prob_path) as src_p:
            prob_ml = src_p.read(1).astype(np.float32)

    logger.info("Generating agreement and method mask rasters...")
    agreement, method_mask = build_agreement_raster(classical, ml)

    logger.info("Generating fused flood mask with rule '%s'...", rule)
    fused = build_fused_flood_raster(classical, ml, rule=rule, prob_ml=prob_ml)

    # Profile for uint8 outputs
    profile.update(
        dtype="uint8",
        nodata=255,
        count=1,
        compress="lzw",
    )

    outputs_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_agreement, "w", **profile) as dst:
        dst.write(agreement, 1)
    with rasterio.open(out_method_mask, "w", **profile) as dst:
        dst.write(method_mask, 1)
    with rasterio.open(out_fused, "w", **profile) as dst:
        dst.write(fused, 1)

    # Verify grid alignment of newly generated rasters
    grid.check_aligned(out_agreement, out_method_mask, out_fused, path_classical)

    # Calculate statistics
    px_area_km2 = (config.PIXEL_SIZE * config.PIXEL_SIZE) / 1e6

    c_flood_px = int((classical == 1).sum())
    m_flood_px = int((ml == 1).sum())
    fused_flood_px = int((fused == 1).sum())

    c_flood_km2 = round(c_flood_px * px_area_km2, 3)
    m_flood_km2 = round(m_flood_px * px_area_km2, 3)
    fused_flood_km2 = round(fused_flood_px * px_area_km2, 3)

    agree_0_px = int((agreement == 0).sum())
    agree_1_px = int((agreement == 1).sum())
    agree_2_px = int((agreement == 2).sum())
    agree_nodata_px = int((agreement == 255).sum())

    ml_avail_px = int((method_mask == 0).sum())
    ml_unavail_px = int((method_mask == 1).sum())

    logger.info("Fusion stats: classical=%.2f km², ml=%.2f km² -> fused=%.2f km² (rule: %s)",
                c_flood_km2, m_flood_km2, fused_flood_km2, rule)
    logger.info("Agreement counts: 0 (neither)=%d, 1 (one)=%d, 2 (both)=%d, nodata=%d",
                agree_0_px, agree_1_px, agree_2_px, agree_nodata_px)

    fusion_payload = {
        "fusion": {
            "rule": rule,
            "rule_description": (
                "classical_plus_ml: SAR classical flood plus high-confidence ML detections; "
                "classical only where ML is unavailable."
                if rule == "classical_plus_ml" else rule
            ),
            "classical_flood_pixels": c_flood_px,
            "classical_flood_km2": c_flood_km2,
            "ml_flood_pixels": m_flood_px,
            "ml_flood_km2": m_flood_km2,
            "fused_flood_pixels": fused_flood_px,
            "fused_flood_km2": fused_flood_km2,
            "agreement_counts": {
                "neither_0": agree_0_px,
                "single_method_1": agree_1_px,
                "both_methods_2": agree_2_px,
                "nodata_255": agree_nodata_px,
            },
            "method_availability": {
                "both_available_pixels": ml_avail_px,
                "ml_unavailable_pixels": ml_unavail_px,
            },
        }
    }

    meta_path = outputs_dir / "meta.json"
    if meta_path.exists():
        update_meta(meta_path, fusion_payload)

    return fusion_payload["fusion"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Classical SAR and Optical ML flood detections")
    parser.add_argument("--rule", default=config.FUSION_RULE, choices=["classical_plus_ml", "intersection", "union"],
                        help=f"Fusion rule (default: {config.FUSION_RULE})")
    parser.add_argument("--force", action="store_true", help="Overwrite existing fused outputs")
    args = parser.parse_args()

    res = run_fusion(rule=args.rule, force=args.force)
    print("\n=== Fusion Execution Complete ===")
    print(f"Rule: {res.get('rule')}")
    print(f"Classical flood: {res.get('classical_flood_km2')} km² ({res.get('classical_flood_pixels')} px)")
    print(f"ML flood:        {res.get('ml_flood_km2')} km² ({res.get('ml_flood_pixels')} px)")
    print(f"Fused flood:     {res.get('fused_flood_km2')} km² ({res.get('fused_flood_pixels')} px)")
    print(f"Agreement:       {res.get('agreement_counts')}")


if __name__ == "__main__":
    main()
