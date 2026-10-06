"""
TerraAlert — Temporal flood detection via ML (Prithvi EO model).

This module implements the MANDATORY pre/post flood detection logic using the
Prithvi EO foundation model. It enforces the hard rule from AGENT_CONTEXT.md:

    flood_ml = water_post AND NOT water_pre AND NOT perm_water

That is: a pixel is a flood only if it is water NOW and was NOT water BEFORE
and is NOT permanent water. This prevents rivers/lakes from being flagged.

Cloud/nodata rule:
    If either the pre-event OR post-event S2 image has nodata (-9999) at a pixel,
    that pixel is set to 255 (unusable) in flood_ml.tif. We never claim a
    detection from a single image.

Outputs:
    outputs/water_pre.tif   — uint8 water mask from pre-event S2
    outputs/water_post.tif  — uint8 water mask from post-event S2
    outputs/flood_ml.tif    — uint8, 1=new flood, 0=not flood, 255=unusable

Meta fields recorded:
    ml_available_fraction   — fraction of pixels with usable (non-255) flood_ml
    model_name              — repo ID of the model that succeeded
    model_version           — weight filename used

FALLBACK RULE:
    If primary model fails → try ML_MODEL_FALLBACK once.
    If that also fails → write flood_ml.tif as all-255,
    set fallbacks.ml = "unavailable: <reason>", exit with clear message.
    NEVER substitute fake output.

CLI:
    python -m backend.ml.temporal [--model NAME] [--device auto|cpu|cuda] [--force]

Sen1Floods11 chip validation:
    On India S2Hand chips, run predict_water (post water mask) against LabelHand.
    Results stored under metrics.ml_water_on_chips.
    THIS IS NOT accuracy on the AOI. See AGENT_CONTEXT rule 5.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import grid
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("ml.temporal")


def _read_s2_array(path: Path) -> np.ndarray:
    """Read a 6-band S2 GeoTIFF as float32 (C, H, W), nodata -9999."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (6, H, W)
    return arr


def _write_uint8_mask(
    mask: np.ndarray,
    path: Path,
    ref_path: Path,
) -> None:
    """Write a uint8 mask aligned to the reference raster's grid."""
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])
    logger.info("Wrote %s (shape %s)", path.name, mask.shape)


def run_ml_detection(
    model_name: str = config.ML_MODEL,
    device: str = config.ML_DEVICE,
    force: bool = False,
    outputs_dir: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full ML pre/post temporal flood detection.

    Steps:
    1. Load s2_pre.tif and s2_post.tif.
    2. Load model (tries primary, then fallback).
    3. predict_water(s2_pre) → water_pre
    4. predict_water(s2_post) → water_post
    5. flood_ml = water_post AND NOT water_pre AND NOT perm_water
    6. Pixels with nodata in EITHER image → 255 in flood_ml.
    7. Save flood_ml.tif, water_pre.tif, water_post.tif.
    8. Record metrics in meta.json.

    Returns
    -------
    dict with 'flood_ml_path', 'ml_available_fraction', 'model_name'
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    if meta_path is None:
        meta_path = outputs_dir / "meta.json"

    out_flood = outputs_dir / "flood_ml.tif"
    out_water_pre = outputs_dir / "water_pre.tif"
    out_water_post = outputs_dir / "water_post.tif"

    if out_flood.exists() and not force:
        logger.info("flood_ml.tif exists. Skipping (use --force to overwrite).")
        return {"flood_ml_path": out_flood, "skipped": True}

    # ── Verify inputs ─────────────────────────────────────────────────────────
    s2_pre_path = outputs_dir / "s2_pre.tif"
    s2_post_path = outputs_dir / "s2_post.tif"
    perm_water_path = outputs_dir / "perm_water.tif"

    missing = [p for p in [s2_pre_path, s2_post_path, perm_water_path] if not p.exists()]
    if missing:
        msg = f"Missing required inputs: {[str(p) for p in missing]}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    grid.check_aligned(s2_pre_path, s2_post_path, perm_water_path)
    logger.info("Grid alignment check passed for ML inputs.")

    # ── Load S2 arrays ────────────────────────────────────────────────────────
    logger.info("Reading s2_pre.tif ...")
    s2_pre = _read_s2_array(s2_pre_path)
    logger.info("Reading s2_post.tif ...")
    s2_post = _read_s2_array(s2_post_path)

    with rasterio.open(perm_water_path) as src:
        perm_water = src.read(1)  # uint8, 1=permanent water

    # ── Combined nodata mask (nodata in EITHER image) ─────────────────────────
    nodata_either = np.any(s2_pre == -9999.0, axis=0) | np.any(s2_post == -9999.0, axis=0)

    # ── Try to load and run model ──────────────────────────────────────────────
    from backend.ml.prithvi import load_model, predict_water

    used_model = None
    last_error = None

    for model_candidate in [model_name, config.ML_MODEL_FALLBACK]:
        if model_candidate == model_name and model_candidate == config.ML_MODEL_FALLBACK:
            # They're the same (shouldn't happen), skip duplicate
            break
        try:
            logger.info("Attempting to load model: %s", model_candidate)
            lightning_model, resolved_device = load_model(model_candidate, device)

            logger.info("Running predict_water on PRE image ...")
            water_pre_mask, _prob_pre = predict_water(
                s2_pre,
                tile=config.ML_TILE,
                overlap=config.ML_OVERLAP,
                batch=1,
                lightning_model=lightning_model,
                device=resolved_device,
            )

            logger.info("Running predict_water on POST image ...")
            water_post_mask, _prob_post = predict_water(
                s2_post,
                tile=config.ML_TILE,
                overlap=config.ML_OVERLAP,
                batch=1,
                lightning_model=lightning_model,
                device=resolved_device,
            )

            del lightning_model
            gc.collect()

            used_model = model_candidate
            break

        except Exception as exc:
            last_error = exc
            logger.warning("Model %s failed: %s. Trying fallback ...", model_candidate, exc)
            gc.collect()

    if used_model is None:
        # Both models failed — write all-255 and record fallback
        H, W = s2_pre.shape[1], s2_pre.shape[2]
        all_unusable = np.full((H, W), 255, dtype=np.uint8)
        _write_uint8_mask(all_unusable, out_flood, s2_pre_path)
        _write_uint8_mask(all_unusable, out_water_pre, s2_pre_path)
        _write_uint8_mask(all_unusable, out_water_post, s2_pre_path)

        reason = f"unavailable: {last_error}"
        update_meta(meta_path, {"fallbacks": {"ml": reason}})
        msg = (
            f"\n[ML FALLBACK] Both models failed. flood_ml.tif written as all-255.\n"
            f"Reason: {reason}\n"
            f"The pipeline can still run using flood_classical.tif.\n"
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        return {
            "flood_ml_path": out_flood,
            "ml_available_fraction": 0.0,
            "model_name": "unavailable",
            "fallback_reason": reason,
        }

    # ── Apply pre/post flood logic ─────────────────────────────────────────────
    # flood_ml = water_post AND NOT water_pre AND NOT perm_water
    # Pixels with nodata in EITHER image → 255
    water_pre_valid = (water_pre_mask == 1)
    water_post_valid = (water_post_mask == 1)
    perm = (perm_water == 1)

    flood_ml = (water_post_valid & ~water_pre_valid & ~perm).astype(np.uint8)
    flood_ml[nodata_either] = 255  # cloud/nodata in either image → unusable

    # Propagate nodata from water_pre and water_post masks (255 = nodata in S2)
    flood_ml[water_pre_mask == 255] = 255
    flood_ml[water_post_mask == 255] = 255

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_uint8_mask(water_pre_mask, out_water_pre, s2_pre_path)
    _write_uint8_mask(water_post_mask, out_water_post, s2_pre_path)
    _write_uint8_mask(flood_ml, out_flood, s2_pre_path)

    # ── Compute available fraction ─────────────────────────────────────────────
    total_pixels = int(flood_ml.size)
    usable_pixels = int(np.sum(flood_ml != 255))
    ml_available_fraction = round(usable_pixels / total_pixels, 4) if total_pixels > 0 else 0.0
    logger.info(
        "ML flood detection complete. Usable pixels: %d / %d (%.1f%%)",
        usable_pixels, total_pixels, ml_available_fraction * 100,
    )

    # ── Record model version and metrics in meta.json ─────────────────────────
    # Derive weight filename from model name
    weight_filename = {
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
        "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11": "sen1floods11_Prithvi_100M.pth",
    }.get(used_model, "unknown")

    update_meta(
        meta_path,
        {
            "model_name": used_model,
            "model_version": weight_filename,
            "ml_available_fraction": ml_available_fraction,
            "fallbacks": {"ml": None},  # None = successfully ran
        },
    )

    return {
        "flood_ml_path": out_flood,
        "ml_available_fraction": ml_available_fraction,
        "model_name": used_model,
        "model_version": weight_filename,
    }


def evaluate_ml_on_chips(
    model_name: str = config.ML_MODEL,
    device: str = config.ML_DEVICE,
    raw_dir: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    max_chips: int = 5,
) -> Dict[str, Any]:
    """Validate predict_water on Sen1Floods11 S2Hand India chips.

    Scores predicted post-water mask against LabelHand labels.
    -1 pixels are always masked. Results recorded under metrics.ml_water_on_chips.

    DISCLAIMER: Sen1Floods11 chips are from other Indian flood events, not
    the Assam 2022 AOI. This validates the model's preprocessing pipeline, not
    the project's flood map accuracy.

    Parameters
    ----------
    model_name : str
        HuggingFace repo ID.
    device : str
        Device for inference.
    raw_dir : Path, optional
        Path to data/raw/ with S2Hand and LabelHand subdirs.
    meta_path : Path, optional
        Path to meta.json.
    max_chips : int
        Number of India chips to evaluate (for time budget).

    Returns
    -------
    dict with per-chip IoU/precision/recall/f1 and aggregate means.
    """
    from backend.ml.prithvi import load_model, predict_water
    from backend.metrics.evaluate import compute_binary_metrics

    if raw_dir is None:
        raw_dir = config.DATA_DIR / "raw"
    if meta_path is None:
        meta_path = config.OUTPUTS_DIR / "meta.json"

    s2_dir = raw_dir / "S2Hand"
    label_dir = raw_dir / "LabelHand"

    if not s2_dir.exists() or not label_dir.exists():
        logger.warning("Sen1Floods11 S2Hand or LabelHand not found in %s", raw_dir)
        update_meta(meta_path, {"metrics": {"ml_water_on_chips": "not run: data not found"}})
        return {}

    india_chips = sorted(s2_dir.glob("India_*.tif"))[:max_chips]
    if not india_chips:
        logger.warning("No India chips found in %s", s2_dir)
        update_meta(meta_path, {"metrics": {"ml_water_on_chips": "not run: no India chips found"}})
        return {}

    logger.info("Loading model %s for chip validation ...", model_name)
    try:
        lightning_model, resolved_device = load_model(model_name, device)
    except Exception as exc:
        logger.error("Could not load model for chip evaluation: %s", exc)
        update_meta(meta_path, {"metrics": {"ml_water_on_chips": f"not run: {exc}"}})
        return {}

    chip_results = []
    for chip_path in india_chips:
        chip_id = chip_path.stem.replace("_S2Hand", "")
        lbl_path = label_dir / f"{chip_id}_LabelHand.tif"
        if not lbl_path.exists():
            logger.warning("Label not found for %s, skipping", chip_id)
            continue

        logger.info("Evaluating chip %s ...", chip_id)
        try:
            # S2Hand chips have 13 bands (all S2 bands); we need bands B2,B3,B4,B8A,B11,B12
            # In Sen1Floods11 S2Hand: 1-indexed bands 2,3,4,9,12,13 → 0-indexed 1,2,3,8,11,12
            with rasterio.open(chip_path) as src:
                n_bands = src.count
                if n_bands >= 13:
                    # Full S2 L1C stack: B1..B13
                    indices = [1, 2, 3, 8, 11, 12]  # 0-indexed
                    all_bands = src.read().astype(np.float32)
                    s2_6band = all_bands[indices, :, :]
                elif n_bands == 6:
                    s2_6band = src.read().astype(np.float32)
                else:
                    logger.warning("Chip %s has %d bands, expected 6 or 13, skipping.", chip_id, n_bands)
                    continue

            with rasterio.open(lbl_path) as src:
                label = src.read(1).astype(np.int16)

            # Run water mask prediction on the post-event chip
            water_mask, _prob = predict_water(
                s2_6band,
                tile=config.ML_TILE,
                overlap=config.ML_OVERLAP,
                batch=1,
                lightning_model=lightning_model,
                device=resolved_device,
            )

            # Score water_mask against LabelHand (1=water, 0=land, -1=nodata)
            # LabelHand has -1 for nodata (cloud), which must be masked per AGENT_CONTEXT rule 5
            metrics = compute_binary_metrics(water_mask, label)
            metrics["chip_id"] = chip_id
            chip_results.append(metrics)
            logger.info("Chip %s: IoU=%.3f P=%.3f R=%.3f F1=%.3f",
                        chip_id, metrics["iou"], metrics["precision"],
                        metrics["recall"], metrics["f1"])

        except Exception as exc:
            logger.warning("Chip %s evaluation failed: %s", chip_id, exc)
            continue

    del lightning_model
    gc.collect()

    if not chip_results:
        update_meta(meta_path, {"metrics": {"ml_water_on_chips": "not run: all chips failed"}})
        return {}

    # Aggregate
    mean_iou = round(float(np.mean([r["iou"] for r in chip_results])), 4)
    mean_p = round(float(np.mean([r["precision"] for r in chip_results])), 4)
    mean_r = round(float(np.mean([r["recall"] for r in chip_results])), 4)
    mean_f1 = round(float(np.mean([r["f1"] for r in chip_results])), 4)

    result = {
        "disclaimer": (
            "Scored on Sen1Floods11 India S2Hand chips, NOT on the Assam 2022 AOI. "
            "These chips are from different flood events in India. "
            "Per AGENT_CONTEXT rule 5, zero accuracy is claimed on the project AOI."
        ),
        "model": model_name,
        "n_chips": len(chip_results),
        "mean_iou": mean_iou,
        "mean_precision": mean_p,
        "mean_recall": mean_r,
        "mean_f1": mean_f1,
        "per_chip": [
            {k: v for k, v in r.items() if k in ("chip_id", "iou", "precision", "recall", "f1")}
            for r in chip_results
        ],
    }

    update_meta(meta_path, {"metrics": {"ml_water_on_chips": result}})
    logger.info(
        "Chip validation complete: n=%d, mean IoU=%.3f, mean F1=%.3f",
        len(chip_results), mean_iou, mean_f1
    )
    return result


def main() -> None:
    """CLI entry point: python -m backend.ml.temporal."""
    parser = argparse.ArgumentParser(
        description="TerraAlert ML temporal flood detection (pre/post Prithvi EO)"
    )
    parser.add_argument(
        "--model",
        default=config.ML_MODEL,
        help=f"HuggingFace repo ID for the model (default: {config.ML_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if flood_ml.tif exists",
    )
    parser.add_argument(
        "--validate-chips",
        action="store_true",
        help="Also run chip-level validation on Sen1Floods11 India S2Hand chips",
    )
    parser.add_argument(
        "--max-chips",
        type=int,
        default=5,
        help="Number of India chips to use for validation (default: 5)",
    )
    args = parser.parse_args()

    print(f"[ML] Model: {args.model}")
    print(f"[ML] Device: {args.device}")

    # Run flood detection
    result = run_ml_detection(
        model_name=args.model,
        device=args.device,
        force=args.force,
    )

    if result.get("skipped"):
        print("[ML] Skipped (outputs exist). Use --force to re-run.")
    elif result.get("fallback_reason"):
        print(f"[ML] FALLBACK: {result['fallback_reason']}")
        print("[ML] flood_ml.tif written as all-255 (unusable).")
        sys.exit(1)
    else:
        print(f"[ML] Success. Model: {result['model_name']}")
        print(f"[ML] Usable fraction: {result['ml_available_fraction']:.1%}")
        print(f"[ML] Outputs: flood_ml.tif, water_pre.tif, water_post.tif")

    # Optionally run chip validation
    if args.validate_chips:
        print("\n[ML] Running chip-level validation on Sen1Floods11 India chips ...")
        chip_result = evaluate_ml_on_chips(
            model_name=args.model,
            device=args.device,
            max_chips=args.max_chips,
        )
        if chip_result:
            print(f"[ML Chips] n={chip_result['n_chips']}, "
                  f"mean IoU={chip_result['mean_iou']:.3f}, "
                  f"mean F1={chip_result['mean_f1']:.3f}")
            print(f"[ML Chips] {chip_result['disclaimer']}")


if __name__ == "__main__":
    main()
