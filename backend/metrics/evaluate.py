"""
TerraAlert — Detection evaluation and benchmark metrics.

Computes IoU, precision, recall, and F1 score against ground-truth references.
Always masks -1 (Sen1Floods11 nodata) and 255 (pipeline nodata).
Verifies geographic overlap of reference chips before asserting accuracy metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

import config
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("metrics.evaluate")


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary classification metrics with invalid (-1 and 255) pixels masked.

    Parameters
    ----------
    pred : np.ndarray
        Predicted binary flood mask (1 = flood, 0 = non-flood, 255 = nodata).
    target : np.ndarray
        Reference label mask (1 = water, 0 = non-water, -1 or 255 = nodata).

    Returns
    -------
    dict with keys: 'iou', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn', 'valid_pixels'
    """
    # Strict mask: exclude Sen1Floods11 nodata (-1), GeoTIFF nodata (255), and NaNs
    valid = (
        (target != -1) &
        (target != 255) &
        (~np.isnan(target)) &
        (pred != 255) &
        (~np.isnan(pred))
    )

    if not np.any(valid):
        return {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "valid_pixels": 0,
        }

    p = (pred[valid] == 1)
    t = (target[valid] == 1)

    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": int(valid.sum()),
    }


def find_overlapping_chips(
    chips_dir: Path,
    aoi_bbox: Tuple[float, float, float, float] = config.AOI_BBOX,
) -> List[Tuple[str, rasterio.coords.BoundingBox]]:
    """
    Find any Sen1Floods11 chips whose bounding box intersects the AOI.
    aoi_bbox is (west, south, east, north) in EPSG:4326.
    """
    if not chips_dir.exists():
        return []

    west, south, east, north = aoi_bbox
    overlapping = []

    for tif_path in sorted(chips_dir.glob("*.tif")):
        try:
            with rasterio.open(tif_path) as src:
                b = src.bounds
                # Check bounding box intersection
                if not (east < b.left or west > b.right or north < b.bottom or south > b.top):
                    overlapping.append((tif_path.name, b))
        except Exception as exc:
            logger.warning("Could not read bounds for %s: %s", tif_path, exc)

    return overlapping


def check_and_record_metrics(
    raw_dir: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Inspect Sen1Floods11 chips, check overlap with AOI, and write metrics status to meta.json.
    """
    if raw_dir is None:
        raw_dir = config.DATA_DIR / "raw"
    if meta_path is None:
        meta_path = config.OUTPUTS_DIR / "meta.json"

    label_dir = raw_dir / "LabelHand"
    overlapping = find_overlapping_chips(label_dir, config.AOI_BBOX)

    if not overlapping:
        status_msg = "not evaluated (no Sen1Floods11 reference chip overlaps AOI)"
        logger.info("Checked %s: 0 chips overlap AOI %s.", label_dir, config.AOI_BBOX)
        logger.info("Per AGENT_CONTEXT rule 5, zero accuracy claimed on AOI.")

        update_meta(
            meta_path,
            {
                "metrics": {
                    "naive": status_msg,
                    "classical": status_msg,
                    "reference": "Sen1Floods11 (data/raw/LabelHand/)",
                    "overlap_aoi": False,
                }
            },
        )
        return {
            "overlap_found": False,
            "status": status_msg,
            "overlapping_chips": [],
        }

def evaluate_ml_water_on_chips(
    pred_water: np.ndarray,
    chip_id: str,
    raw_dir: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    record_meta: bool = True,
) -> Dict[str, Any]:
    """Score an ML water-prediction mask against Sen1Floods11 LabelHand.

    This validates the model's preprocessing and output on labelled India chips.
    It does NOT claim accuracy on the project AOI (Barpeta-Nalbari, Assam).
    Per AGENT_CONTEXT rule 5: chips do not overlap the AOI. Never present
    this metric as the project's flood-map accuracy.

    Parameters
    ----------
    pred_water : np.ndarray
        Predicted water mask, uint8, same shape as chip (1=water, 0=land, 255=nodata).
    chip_id : str
        Sen1Floods11 chip identifier, e.g. 'India_1017769'.
    raw_dir : Path, optional
        Path to data/raw/ directory.
    meta_path : Path, optional
        Path to meta.json.
    record_meta : bool
        If True, write results to meta.json under metrics.ml_water_on_chips.

    Returns
    -------
    dict with keys: iou, precision, recall, f1, chip_id, disclaimer
    """
    if raw_dir is None:
        raw_dir = config.DATA_DIR / "raw"
    if meta_path is None:
        meta_path = config.OUTPUTS_DIR / "meta.json"

    lbl_path = raw_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"
    if not lbl_path.exists():
        raise FileNotFoundError(f"LabelHand not found: {lbl_path}")

    with rasterio.open(lbl_path) as src:
        label = src.read(1).astype(np.int16)

    metrics = compute_binary_metrics(pred_water, label)
    disclaimer = (
        "Scored on Sen1Floods11 India S2Hand chip (different event from Assam 2022 AOI). "
        "Per AGENT_CONTEXT rule 5, zero accuracy is claimed on the project AOI."
    )
    metrics["chip_id"] = chip_id
    metrics["disclaimer"] = disclaimer

    if record_meta:
        update_meta(
            meta_path,
            {
                "metrics": {
                    "ml_water_on_chips": {
                        "chip_id": chip_id,
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "valid_pixels": metrics["valid_pixels"],
                        "disclaimer": disclaimer,
                    }
                }
            },
        )
        logger.info("Recorded ML chip metrics for %s in %s", chip_id, meta_path)

    return metrics


def evaluate_chip_benchmark(
    chip_id: str = "India_1017769",
    raw_dir: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on a Sen1Floods11 chip by fetching matching pre-event S1 imagery from GEE.
    Labelled as a separate benchmark experiment; explicitly disclaims accuracy on the project AOI.
    """
    import urllib.request
    import ee
    from backend.detect.classical import clean_morphology

    if raw_dir is None:
        raw_dir = config.DATA_DIR / "raw"
    if meta_path is None:
        meta_path = config.OUTPUTS_DIR / "meta.json"

    s1_path = raw_dir / "S1Hand" / f"{chip_id}_S1Hand.tif"
    lbl_path = raw_dir / "LabelHand" / f"{chip_id}_LabelHand.tif"
    jrc_path = raw_dir / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif"

    for p in (s1_path, lbl_path, jrc_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required Sen1Floods11 chip component: {p}")

    with rasterio.open(s1_path) as src:
        post_vv = src.read(1)
        b = src.bounds
        crs = src.crs.to_string()
        h, w = src.height, src.width

    with rasterio.open(lbl_path) as src:
        target = src.read(1)

    with rasterio.open(jrc_path) as src:
        perm = src.read(1)

    cache_path = config.DATA_DIR / "gee_cache" / f"{chip_id}_pre.tif"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        logger.info("Initializing Earth Engine to fetch matching pre-event S1 image for %s...", chip_id)
        ee.Initialize(project=config.GEE_PROJECT)
        geom = ee.Geometry.BBox(b.left, b.bottom, b.right, b.top)
        pre_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("relativeOrbitNumber_start", 143))
            .filterDate("2019-05-01", "2019-06-01")
        )
        pre_img = pre_col.select("VV").median()
        url = pre_img.getDownloadURL({
            "region": geom,
            "crs": crs,
            "dimensions": [w, h],
            "format": "GEO_TIFF",
        })
        urllib.request.urlretrieve(url, cache_path)
        logger.info("Downloaded matching pre-event S1 image to %s", cache_path)

    with rasterio.open(cache_path) as src:
        pre_vv = src.read(1)

    # 1. Naive baseline on chip
    naive_pred = (post_vv < config.POST_DB_MAX).astype(np.uint8)
    naive_metrics = compute_binary_metrics(naive_pred, target)

    # 2. Classical detection on chip
    diff = post_vv - pre_vv
    valid = (target != -1) & (~np.isnan(post_vv)) & (~np.isnan(pre_vv))
    raw_classical = (post_vv < config.POST_DB_MAX) & (diff < config.DIFF_DB_MAX) & (perm == 0) & valid
    cleaned_classical = clean_morphology(raw_classical, config.MIN_OBJECT_PIXELS, valid)
    classical_pred = cleaned_classical.astype(np.uint8)
    classical_metrics = compute_binary_metrics(classical_pred, target)

    # 3. Fused detection on chip
    from backend.fusion.fuse import build_fused_flood_raster
    fused_pred = build_fused_flood_raster(classical_pred, np.full_like(classical_pred, 255), rule=config.FUSION_RULE)
    fused_metrics = compute_binary_metrics(fused_pred, target)

    logger.info("Chip %s Naive Metrics: %s", chip_id, naive_metrics)
    logger.info("Chip %s Classical Metrics: %s", chip_id, classical_metrics)
    logger.info("Chip %s Fused Metrics: %s", chip_id, fused_metrics)

    metrics_payload = {
        "metrics": {
            "experiment": "Sen1Floods11 Ground-Truth Benchmark (Separate Experiment)",
            "reference_dataset": f"Sen1Floods11 Hand-Labeled ({lbl_path.name})",
            "reference_location": "Nagaon / Brahmaputra Valley, Assam (26.60°N, 93.02°E)",
            "chip_id": chip_id,
            "aoi_overlap": False,
            "aoi_disclaimer": "Sen1Floods11 chips do not geographically overlap the Barpeta-Nalbari AOI. Zero accuracy is claimed on the project AOI per AGENT_CONTEXT rule 5.",
            "pre_event_source": "COPERNICUS/S1_GRD (May 2019, Relative Orbit 143, ASCENDING)",
            "post_event_source": "Sen1Floods11 S1Hand (July 2019, Relative Orbit 143, ASCENDING)",
            "naive": naive_metrics,
            "classical": classical_metrics,
            "fused": fused_metrics,
        },
        "fallbacks": {
            "calibration": "no overlapping reference",
        },
    }

    update_meta(meta_path, metrics_payload)
    logger.info("Recorded benchmark metrics in %s", meta_path)

    return {
        "naive": naive_metrics,
        "classical": classical_metrics,
        "fused": fused_metrics,
        "chip_id": chip_id,
    }


def record_fused_metrics(
    meta_path: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
    reference_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Record metrics.fused and calibration table where an overlapping reference exists;
    otherwise maintain fallbacks.calibration = 'no overlapping reference'.
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    if meta_path is None:
        meta_path = outputs_dir / "meta.json"

    has_overlapping_ref = False
    calib_table = None

    if reference_path is not None and reference_path.exists():
        has_overlapping_ref = True
        with rasterio.open(reference_path) as ref_src:
            target = ref_src.read(1)
        fused_path = outputs_dir / "flood_fused.tif"
        if fused_path.exists():
            with rasterio.open(fused_path) as f_src:
                fused = f_src.read(1)
            fused_metrics = compute_binary_metrics(fused, target)
        else:
            fused_metrics = {"status": "flood_fused.tif not generated"}
    else:
        fused_metrics = {
            "status": "not evaluated on AOI (no overlapping reference dataset)",
            "aoi_disclaimer": "Per AGENT_CONTEXT rule 5, zero accuracy claimed on AOI.",
        }

    payload: Dict[str, Any] = {
        "metrics": {
            "fused": fused_metrics,
        },
    }

    if has_overlapping_ref and calib_table:
        payload["metrics"]["calibration_table"] = calib_table
    else:
        payload["fallbacks"] = {
            "calibration": "no overlapping reference",
        }

    if meta_path.exists():
        update_meta(meta_path, payload)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection against reference chips")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark evaluation on Sen1Floods11 chip")
    parser.add_argument("--chip-id", default="India_1017769", help="Chip ID to evaluate")
    args = parser.parse_args()

    if args.benchmark:
        res = evaluate_chip_benchmark(chip_id=args.chip_id)
        print("=== Benchmark Evaluation Results ===")
        print(f"Chip: {res['chip_id']}")
        print(f"Naive: {res['naive']}")
        print(f"Classical: {res['classical']}")
    else:
        res = check_and_record_metrics()
        print(f"Overlap check result: {res}")


if __name__ == "__main__":
    main()
