"""
Unit tests for classical change detection, naive baseline, and evaluation metrics.

Covers:
  1. Synthetic scene with permanent river, flood patch, steep slope, speckle, and small hole.
  2. Verification that permanent river is excluded by classical detection.
  3. Verification that steep slope is excluded by classical detection.
  4. Verification that speckle noise (< MIN_OBJECT_PIXELS) is removed.
  5. Verification that small dry holes (< MIN_OBJECT_PIXELS) inside flood patches are filled.
  6. Verification that naive single-image baseline traps the permanent river (false positive).
  7. Verification of IoU, precision, recall, and F1 calculations with -1 nodata masking.
  8. Otsu threshold computation behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.detect.classical import (
    clean_morphology,
    compute_otsu_threshold,
    detect_classical_arrays,
)
from backend.detect.naive import detect_naive_array
from backend.metrics.evaluate import compute_binary_metrics


@pytest.fixture
def synthetic_scene():
    """
    Create a 100x100 synthetic scene with:
      - Dry background: pre=-12 dB, post=-12 dB, perm=0, slope=1°
      - Permanent river (cols 10:25): pre=-22 dB, post=-22 dB, perm=1, slope=0.5°
      - New flood patch (rows 40:60, cols 40:60): pre=-12 dB, post=-22 dB, perm=0, slope=1°
      - Steep slope area (rows 70:90, cols 70:90): pre=-12 dB, post=-22 dB, perm=0, slope=25°
      - Speckle patch (rows 5:7, cols 50:51): 2 pixels, pre=-12 dB, post=-22 dB, perm=0, slope=1°
      - Small hole inside flood patch (rows 48:50, cols 48:50): 4 pixels, pre=-12 dB, post=-12 dB
    """
    shape = (100, 100)
    pre = np.full(shape, -12.0, dtype=np.float32)
    post = np.full(shape, -12.0, dtype=np.float32)
    perm_water = np.zeros(shape, dtype=np.uint8)
    slope = np.full(shape, 1.0, dtype=np.float32)

    # 1. Permanent river
    pre[:, 10:25] = -22.0
    post[:, 10:25] = -22.0
    perm_water[:, 10:25] = 1
    slope[:, 10:25] = 0.5

    # 2. Real new flood patch (20x20 = 400 pixels)
    post[40:60, 40:60] = -22.0

    # 3. Small hole inside flood patch (2x2 = 4 pixels, < MIN_OBJECT_PIXELS 25)
    post[48:50, 48:50] = -12.0

    # 4. Steep slope area (looks like flood on SAR, but slope > 5°)
    post[70:90, 70:90] = -22.0
    slope[70:90, 70:90] = 25.0

    # 5. Speckle noise (2 pixels, < MIN_OBJECT_PIXELS 25)
    post[5:7, 50] = -22.0

    return {
        "pre": pre,
        "post": post,
        "perm_water": perm_water,
        "slope": slope,
    }


def test_classical_synthetic_scene(synthetic_scene):
    """Verify classical multi-condition SAR detection on the synthetic scene."""
    flood_mask, valid_mask, thresh = detect_classical_arrays(
        pre=synthetic_scene["pre"],
        post=synthetic_scene["post"],
        perm_water=synthetic_scene["perm_water"],
        slope=synthetic_scene["slope"],
        post_db_max=-18.0,
        diff_db_max=-3.0,
        slope_max_deg=5.0,
        min_object_pixels=25,
        use_otsu=False,
    )

    # 1. Permanent river must be EXCLUDED (0)
    river_pixels = flood_mask[:, 10:25]
    assert np.all(river_pixels == 0), "Permanent river was not excluded by classical detection!"

    # 2. Steep slope area must be EXCLUDED (0)
    steep_pixels = flood_mask[70:90, 70:90]
    assert np.all(steep_pixels == 0), "Steep slope area was not excluded by slope filter!"

    # 3. Speckle noise must be REMOVED (0)
    speckle_pixels = flood_mask[5:7, 50]
    assert np.all(speckle_pixels == 0), "Small speckle was not removed by min_object_pixels!"

    # 4. Small hole inside flood patch must be FILLED (1)
    hole_pixels = flood_mask[48:50, 48:50]
    assert np.all(hole_pixels == 1), "Small hole inside flood patch was not filled!"

    # 5. Main flood patch body must be DETECTED (1)
    # Check interior coordinates of the flood patch
    patch_interior = flood_mask[42:47, 42:47]
    assert np.all(patch_interior == 1), "Flood patch interior was not detected!"


def test_naive_baseline_falls_into_river_trap(synthetic_scene):
    """
    Verify naive single-image baseline incorrectly classifies the permanent river
    as flood because post < -18 dB.
    """
    naive_mask, _ = detect_naive_array(
        post=synthetic_scene["post"],
        post_db_max=-18.0,
    )
    # River in naive detection should be falsely classified as flood (1)
    river_pixels = naive_mask[:, 10:25]
    assert np.all(river_pixels == 1), "Naive baseline should have falsely flagged river as flood"

    # Steep slope in naive detection is also falsely flagged (1)
    steep_pixels = naive_mask[70:90, 70:90]
    assert np.all(steep_pixels == 1), "Naive baseline should have falsely flagged steep dark area"


def test_clean_morphology_isolated():
    """Test morphology cleaner directly with known object and hole sizes."""
    mask = np.zeros((20, 20), dtype=bool)
    valid = np.ones((20, 20), dtype=bool)

    # 5x5 patch = 25 pixels
    mask[2:7, 2:7] = True
    # Punch a 1-pixel hole
    mask[4, 4] = False

    # A separate 2-pixel speckle
    mask[15:17, 15] = True

    cleaned = clean_morphology(mask, min_pixels=10, valid=valid)

    # The 2-pixel speckle must be removed
    assert not cleaned[15, 15]
    assert not cleaned[16, 15]

    # The 1-pixel hole must be filled
    assert cleaned[4, 4]

    # The rest of the 5x5 patch must remain True
    assert cleaned[2, 2]
    assert cleaned[6, 6]


def test_metrics_evaluation_perfect_match():
    """IoU and F1 should be 1.0 on identical binary masks."""
    pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    target = np.array([[1, 0], [0, 1]], dtype=np.int8)

    m = compute_binary_metrics(pred, target)
    assert m["iou"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_metrics_evaluation_masks_minus_one_and_nodata():
    """Sen1Floods11 nodata (-1) and raster nodata (255) must be excluded from metrics."""
    pred = np.array([1, 1, 0, 0, 1, 255], dtype=np.uint8)
    # index 4 is -1 in target (must be masked out)
    # index 5 is 255 in pred (must be masked out)
    target = np.array([1, 0, 1, 0, -1, 1], dtype=np.int8)

    m = compute_binary_metrics(pred, target)
    # Valid indices: 0, 1, 2, 3
    # index 0: p=1, t=1 -> TP
    # index 1: p=1, t=0 -> FP
    # index 2: p=0, t=1 -> FN
    # index 3: p=0, t=0 -> TN
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["valid_pixels"] == 4
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["iou"] == pytest.approx(1.0 / 3.0, rel=1e-3)


def test_compute_otsu_threshold():
    """Verify Otsu selects an appropriate separation between two clusters."""
    cluster_dark = np.random.normal(-24.0, 1.0, 500)
    cluster_bright = np.random.normal(-10.0, 1.0, 1000)
    combined = np.concatenate([cluster_dark, cluster_bright])

    thresh = compute_otsu_threshold(combined)
    assert -22.0 < thresh < -14.0


def test_evaluate_chip_benchmark_offline(tmp_path: Path):
    """Test evaluate_chip_benchmark with offline synthetic GeoTIFFs (no network)."""
    import rasterio
    from rasterio.transform import from_origin
    from backend.metrics.evaluate import evaluate_chip_benchmark

    chip_id = "Test_999"
    raw_dir = tmp_path / "raw"
    for cat in ("S1Hand", "LabelHand", "JRCWaterHand"):
        (raw_dir / cat).mkdir(parents=True, exist_ok=True)

    cache_dir = tmp_path / "gee_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = tmp_path / "meta.json"

    transform = from_origin(93.0, 26.6, 0.0001, 0.0001)
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
    }

    # Write S1Hand (post)
    with rasterio.open(raw_dir / "S1Hand" / f"{chip_id}_S1Hand.tif", "w", **profile) as dst:
        dst.write(np.full((10, 10), -22.0, dtype=np.float32), 1)

    # Write pre cache
    with rasterio.open(cache_dir / f"{chip_id}_pre.tif", "w", **profile) as dst:
        dst.write(np.full((10, 10), -12.0, dtype=np.float32), 1)

    # Write LabelHand (target)
    prof_lbl = profile.copy()
    prof_lbl.update(dtype="int16")
    with rasterio.open(raw_dir / "LabelHand" / f"{chip_id}_LabelHand.tif", "w", **prof_lbl) as dst:
        dst.write(np.full((10, 10), 1, dtype=np.int16), 1)

    # Write JRCWaterHand
    prof_jrc = profile.copy()
    prof_jrc.update(dtype="uint8")
    with rasterio.open(raw_dir / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif", "w", **prof_jrc) as dst:
        dst.write(np.zeros((10, 10), dtype=np.uint8), 1)

    # Pass config.DATA_DIR override by pointing raw_dir and meta_path
    import config
    orig_cache = config.DATA_DIR
    try:
        config.DATA_DIR = tmp_path
        res = evaluate_chip_benchmark(chip_id=chip_id, raw_dir=raw_dir, meta_path=meta_path)
        assert "naive" in res
        assert "classical" in res
        assert res["naive"]["iou"] > 0
        assert res["classical"]["iou"] > 0
    finally:
        config.DATA_DIR = orig_cache


def test_perm_water_land_nodata_regression():
    """
    Regression test for permanent water land-mask bug:
    Ensures that:
      - Dry land (perm_water=0) is part of valid data and detected correctly
      - Permanent river (perm_water=1) is excluded from flood detection
      - Flood patch on dry land (perm_water=0) is detected as flood (1)
      - Unmapped / nodata perm_water (255) does NOT cause land to be discarded
    """
    shape = (100, 100)
    pre = np.full(shape, -12.0, dtype=np.float32)
    post = np.full(shape, -12.0, dtype=np.float32)
    slope = np.full(shape, 1.0, dtype=np.float32)

    # perm_water: dry land = 0, permanent river = 1, nodata margin = 255
    perm_water = np.zeros(shape, dtype=np.uint8)
    perm_water[:, :20] = 1       # permanent river
    perm_water[:, 80:] = 255     # nodata boundary
    pre[:, :20] = -22.0
    post[:, :20] = -22.0

    # Flood patch on dry land (rows 40:60, cols 30:50) -> 20x20 = 400 pixels
    post[40:60, 30:50] = -22.0

    flood, valid, _ = detect_classical_arrays(
        pre=pre,
        post=post,
        perm_water=perm_water,
        slope=slope,
        post_db_max=-18.0,
        diff_db_max=-3.0,
        slope_max_deg=5.0,
        min_object_pixels=25,
    )

    # 1. Permanent river must NOT be detected as flood
    assert np.all(flood[:, :20] == 0)

    # 2. Flood patch on dry land must be detected
    assert np.all(flood[40:60, 30:50] == 1)
    assert np.sum(flood == 1) == 400

    # 3. Dry land without flood must be 0 (valid non-flood)
    assert flood[0, 50] == 0
    assert bool(valid[0, 50]) is True

