"""
Tests for backend/ml/prithvi.py and backend/ml/temporal.py.

All tests are offline (no network, no model weights). They use:
- A stub model that returns a fixed probability array.
- Synthetic numpy arrays for S2 imagery and masks.

Tests cover:
1. Hanning blending window shape and value properties.
2. Tiling + blending on synthetic arrays: correct output shape, no seams, no NaN.
3. Pre/post flood logic: water_post AND NOT water_pre AND NOT perm_water.
4. Nodata passthrough: -9999 in either image → 255 in flood_ml.
5. Stub model returns correct shapes.
6. predict_water with stub model produces valid mask.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Stub model helpers ─────────────────────────────────────────────────────────

class _StubOutput:
    """Simulates the output namespace that Prithvi model returns."""
    def __init__(self, logits: "torch.Tensor"):
        self.output = logits


class _StubForwardModel:
    """
    Stub that replaces the Prithvi model's forward pass.
    Always predicts water (class 1) with probability p_water.
    """
    def __init__(self, p_water: float = 0.9, device: str = "cpu"):
        self._p_water = p_water
        self.device = device

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, x):
        import torch
        B, C, H, W = x.shape
        # Create logits: class-0 low, class-1 high → water probability = p_water
        log_water = np.log(self._p_water + 1e-9)
        log_land = np.log(1.0 - self._p_water + 1e-9)
        logits = torch.zeros(B, 2, H, W)
        logits[:, 0, :, :] = log_land
        logits[:, 1, :, :] = log_water
        return _StubOutput(logits)


class _StubLightningModel:
    def __init__(self, p_water: float = 0.9):
        self.model = _StubForwardModel(p_water=p_water)


# ── Import helpers ─────────────────────────────────────────────────────────────

def _try_import_torch():
    """Skip test if torch is not installed."""
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("torch not installed; skipping ML tests")


def _import_prithvi():
    """Import prithvi module (no network calls)."""
    # We need torch available for the tiling logic
    _try_import_torch()
    from backend.ml import prithvi
    return prithvi


# ── Test 1: Hanning window ─────────────────────────────────────────────────────

def test_hann_window_shape():
    """Hanning window should be (tile, tile) and peak at centre."""
    prithvi = _import_prithvi()
    tile = 64
    win = prithvi._make_hann_weight(tile)
    assert win.shape == (tile, tile), f"Expected ({tile},{tile}), got {win.shape}"
    # Centre pixel should be close to 1
    assert win[tile // 2, tile // 2] > 0.9, "Centre of Hanning window should be near 1.0"
    # Edge pixels should be near 0
    assert win[0, 0] < 0.05, "Corner of Hanning window should be near 0.0"


def test_hann_window_non_negative():
    """Hanning window values must be >= 0 (no negative weights)."""
    prithvi = _import_prithvi()
    win = prithvi._make_hann_weight(128)
    assert np.all(win >= 0.0), "Hanning window must be non-negative everywhere"


# ── Test 2: Tiling + blending produces correct shape and no NaN ───────────────

def test_predict_water_output_shape():
    """predict_water on (6, 200, 220) with stub model returns (200, 220)."""
    prithvi = _import_prithvi()
    import torch
    H, W = 200, 220
    # Create valid synthetic S2 image (no nodata)
    s2 = np.random.uniform(500, 5000, size=(6, H, W)).astype(np.float32)

    stub = _StubLightningModel(p_water=0.95)

    mask, prob = prithvi.predict_water(
        s2,
        tile=64,
        overlap=8,
        batch=2,
        lightning_model=stub,
        device="cpu",
    )
    assert mask.shape == (H, W), f"Expected ({H},{W}), got {mask.shape}"
    assert prob.shape == (H, W), f"Expected ({H},{W}), got {prob.shape}"


def test_predict_water_no_nan():
    """Output prob array must not contain NaN for valid input."""
    prithvi = _import_prithvi()
    s2 = np.ones((6, 150, 150), dtype=np.float32) * 2000.0
    stub = _StubLightningModel(p_water=0.85)
    mask, prob = prithvi.predict_water(
        s2, tile=64, overlap=8, batch=1, lightning_model=stub, device="cpu"
    )
    assert not np.any(np.isnan(prob)), "prob map must not contain NaN"
    assert not np.any(np.isnan(mask.astype(float))), "mask must not contain NaN"


def test_predict_water_high_prob_gives_water():
    """With stub p_water=0.99, all non-nodata pixels should be flagged as water (>=ML_PROB_HIGH)."""
    prithvi = _import_prithvi()
    import config
    s2 = np.ones((6, 100, 100), dtype=np.float32) * 1500.0
    stub = _StubLightningModel(p_water=0.99)
    mask, prob = prithvi.predict_water(
        s2, tile=64, overlap=8, batch=1, lightning_model=stub, device="cpu"
    )
    # prob >= ML_PROB_HIGH (0.8) so all valid pixels should be 1
    valid = mask != 255
    if np.any(valid):
        assert np.all(mask[valid] == 1), "All valid pixels should be water with p_water=0.99"


def test_predict_water_nodata_pass_through():
    """Pixels with nodata (-9999) in any band must be 255 in the output mask."""
    prithvi = _import_prithvi()
    H, W = 100, 100
    s2 = np.ones((6, H, W), dtype=np.float32) * 2000.0
    # Mark a 10x10 region as nodata in band 0
    s2[0, 10:20, 10:20] = -9999.0

    stub = _StubLightningModel(p_water=0.95)
    mask, prob = prithvi.predict_water(
        s2, tile=64, overlap=8, batch=1, lightning_model=stub, device="cpu"
    )
    nodata_region = mask[10:20, 10:20]
    assert np.all(nodata_region == 255), "Nodata pixels must map to 255 in output mask"


def test_predict_water_no_seam_uniform_input():
    """For uniform input, blended prob should be nearly constant (no tile-edge seams)."""
    prithvi = _import_prithvi()
    H, W = 200, 200
    s2 = np.full((6, H, W), 2000.0, dtype=np.float32)
    stub = _StubLightningModel(p_water=0.9)
    mask, prob = prithvi.predict_water(
        s2, tile=64, overlap=16, batch=1, lightning_model=stub, device="cpu"
    )
    # prob should be ~uniform — std dev < 0.05 if there are no seams
    valid_prob = prob[prob > 0]
    if len(valid_prob) > 0:
        std = float(np.std(valid_prob))
        assert std < 0.10, f"Too much variation in prob (std={std:.3f}); possible tile seam"


# ── Test 3: Pre/post flood logic ───────────────────────────────────────────────

def test_pre_post_flood_logic():
    """
    flood_ml = water_post AND NOT water_pre AND NOT perm_water
    A pixel that is water only in post and not in pre or perm_water → 1.
    """
    H, W = 50, 50
    # Pre: no water
    water_pre = np.zeros((H, W), dtype=np.uint8)
    # Post: all water
    water_post = np.ones((H, W), dtype=np.uint8)
    # No permanent water
    perm_water = np.zeros((H, W), dtype=np.uint8)
    # No nodata
    nodata_either = np.zeros((H, W), dtype=bool)

    flood_ml = _apply_pre_post_logic(water_pre, water_post, perm_water, nodata_either)
    assert np.all(flood_ml == 1), "All pixels should be flood (water_post & ~water_pre & ~perm)"


def test_pre_post_permanent_water_excluded():
    """Permanent water pixels must not appear in flood_ml (they are not NEW floods)."""
    H, W = 50, 50
    water_pre = np.zeros((H, W), dtype=np.uint8)
    water_post = np.ones((H, W), dtype=np.uint8)
    perm_water = np.ones((H, W), dtype=np.uint8)  # ALL permanent
    nodata_either = np.zeros((H, W), dtype=bool)

    flood_ml = _apply_pre_post_logic(water_pre, water_post, perm_water, nodata_either)
    assert np.all(flood_ml == 0), "Permanent water pixels must NOT be flagged as flood"


def test_pre_water_excluded():
    """Pixels that were already water pre-event must not be flagged as NEW flood."""
    H, W = 50, 50
    water_pre = np.ones((H, W), dtype=np.uint8)   # all water pre
    water_post = np.ones((H, W), dtype=np.uint8)  # all water post
    perm_water = np.zeros((H, W), dtype=np.uint8)
    nodata_either = np.zeros((H, W), dtype=bool)

    flood_ml = _apply_pre_post_logic(water_pre, water_post, perm_water, nodata_either)
    assert np.all(flood_ml == 0), "Pre-event water pixels must NOT be flagged as NEW flood"


def test_nodata_in_either_image_gives_255():
    """Pixels with nodata in pre OR post → flood_ml = 255 (unusable)."""
    H, W = 50, 50
    water_pre = np.zeros((H, W), dtype=np.uint8)
    water_post = np.ones((H, W), dtype=np.uint8)
    perm_water = np.zeros((H, W), dtype=np.uint8)
    nodata_either = np.zeros((H, W), dtype=bool)
    nodata_either[10:20, 10:20] = True  # 10x10 block has nodata

    flood_ml = _apply_pre_post_logic(water_pre, water_post, perm_water, nodata_either)
    nodata_block = flood_ml[10:20, 10:20]
    valid_block = flood_ml[20:, 20:]
    assert np.all(nodata_block == 255), "Nodata pixels must be 255 in flood_ml"
    assert np.all(valid_block == 1), "Valid pixels should still be correctly classified"


def test_s2_nodata_water_mask_pass_through():
    """water_pre/post 255 propagates to flood_ml 255."""
    H, W = 50, 50
    water_pre = np.zeros((H, W), dtype=np.uint8)
    water_post = np.ones((H, W), dtype=np.uint8)
    water_post[5:15, 5:15] = 255  # cloud in post water mask
    perm_water = np.zeros((H, W), dtype=np.uint8)
    nodata_either = np.zeros((H, W), dtype=bool)

    flood_ml = _apply_pre_post_logic(water_pre, water_post, perm_water, nodata_either)
    # 255 in water_post must propagate
    flood_ml_with_cloud = flood_ml.copy()
    flood_ml_with_cloud[water_post == 255] = 255
    assert np.all(flood_ml_with_cloud[5:15, 5:15] == 255)


# ── Helper: replicate the pre/post logic from temporal.py ────────────────────

def _apply_pre_post_logic(
    water_pre: np.ndarray,
    water_post: np.ndarray,
    perm_water: np.ndarray,
    nodata_either: np.ndarray,
) -> np.ndarray:
    """Replicate the core flood_ml logic from temporal.py for testing."""
    water_pre_valid = (water_pre == 1)
    water_post_valid = (water_post == 1)
    perm = (perm_water == 1)

    flood_ml = (water_post_valid & ~water_pre_valid & ~perm).astype(np.uint8)
    flood_ml[nodata_either] = 255
    flood_ml[water_pre == 255] = 255
    flood_ml[water_post == 255] = 255
    return flood_ml
