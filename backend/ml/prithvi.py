"""
TerraAlert — Prithvi EO model loader and water-mask inference.

Model contract (read from HuggingFace repos before coding):
  Primary:  ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11
  Fallback: ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11

Input band order (EXACT from config.yaml backbone_bands):
  [0] BLUE   = Sentinel-2 B2
  [1] GREEN  = Sentinel-2 B3
  [2] RED    = Sentinel-2 B4
  [3] NIR_NARROW = Sentinel-2 B8A
  [4] SWIR_1 = Sentinel-2 B11
  [5] SWIR_2 = Sentinel-2 B12

Normalisation (EXACT from config.yaml constant_scale = 0.0001):
  pixel_value = raw_DN * 0.0001    (i.e. divide by 10000; brings 0–10000 → 0.0–1.0)
  No per-band mean/std subtraction; model trained with only constant scaling.

Input size:
  512×512 pixels per tile (img_size used in inference.py).

Output classes:
  2 classes — class 0 = non-water, class 1 = water.
  Softmax channel 1 = water probability.

Weight caching:
  Weights are downloaded via huggingface_hub.hf_hub_download() and cached in
  data/models/ (gitignored). On subsequent runs the local cache is used.

Tiling with overlap blending:
  For images larger than ML_TILE, tiles are extracted with ML_OVERLAP stride
  and predictions are blended in the overlap region using linear weighting
  (hanning-window cosine taper). This eliminates tile-edge seams.

Dependencies (requirements-ml.txt):
  torch>=2.4.0
  terratorch>=0.99
  huggingface_hub>=0.24

FALLBACK RULE:
  If model loading or inference fails, the caller (temporal.py) catches the
  exception and tries ML_MODEL_FALLBACK. If that also fails, flood_ml.tif is
  written as all-255 and fallbacks.ml is recorded.
"""
from __future__ import annotations

import gc
import importlib
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure project root on path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logger = logging.getLogger("ml.prithvi")

# ── Constants ─────────────────────────────────────────────────────────────────
CONSTANT_SCALE: float = 0.0001          # raw DN → reflectance (from config.yaml)
NODATA_IN: float = -9999.0              # nodata value in s2_*.tif
NODATA_REPLACE: float = 0.0            # fill nodata before model (from config no_data_replace)
NUM_CLASSES: int = 2                    # 0=non-water, 1=water
MODELS_DIR: Path = config.DATA_DIR / "models"  # weight cache, gitignored

# Map of repo_id → weight filename (verified from HuggingFace siblings API)
_WEIGHT_FILES: dict[str, str] = {
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11": "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
    "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11": "sen1floods11_Prithvi_100M.pth",
}

_CONFIG_FILES: dict[str, str] = {
    "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11": "config.yaml",
    "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11": "config.yaml",
}


def _require_torch():
    """Import torch or raise ImportError with a clear message."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Install requirements-ml.txt to use ML inference:\n"
            "  pip install -r requirements-ml.txt"
        ) from exc


def _require_terratorch():
    """Import terratorch or raise ImportError with a clear message."""
    try:
        import terratorch  # noqa: F401
        from terratorch.models.backbones.prithvi_model_factory import LightningInferenceModel
        return LightningInferenceModel
    except ImportError as exc:
        raise ImportError(
            "terratorch is not installed. Install requirements-ml.txt:\n"
            "  pip install -r requirements-ml.txt"
        ) from exc


def _require_hf_hub():
    """Import huggingface_hub or raise ImportError with a clear message."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        return hf_hub_download, snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Install requirements-ml.txt:\n"
            "  pip install -r requirements-ml.txt"
        ) from exc


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on availability and config."""
    torch = _require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _download_weights(repo_id: str) -> Tuple[Path, Path]:
    """Download model weights and config to MODELS_DIR cache.

    Returns (config_path, weights_path).
    """
    hf_hub_download, _ = _require_hf_hub()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    weight_file = _WEIGHT_FILES.get(repo_id)
    config_file = _CONFIG_FILES.get(repo_id, "config.yaml")
    if weight_file is None:
        raise ValueError(f"Unknown model repo_id: {repo_id!r}. Add it to _WEIGHT_FILES.")

    # Local cache paths
    safe_name = repo_id.replace("/", "__")
    local_dir = MODELS_DIR / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    local_weights = local_dir / weight_file
    local_config = local_dir / config_file

    if not local_weights.exists():
        logger.info("Downloading weights for %s to %s ...", repo_id, local_dir)
        hf_hub_download(repo_id=repo_id, filename=weight_file, local_dir=str(local_dir))
        logger.info("Weights saved to %s", local_weights)
    else:
        logger.info("Using cached weights at %s", local_weights)

    if not local_config.exists():
        logger.info("Downloading config for %s ...", repo_id)
        hf_hub_download(repo_id=repo_id, filename=config_file, local_dir=str(local_dir))

    return local_config, local_weights


def load_model(name: str = config.ML_MODEL, device: str = "auto"):
    """Load a Prithvi flood-segmentation model with weight caching.

    Parameters
    ----------
    name : str
        HuggingFace repo ID, e.g. 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11'.
    device : str
        'auto' (cuda if available, else cpu), 'cuda', or 'cpu'.

    Returns
    -------
    lightning_model : LightningInferenceModel
        A terratorch LightningInferenceModel ready for eval-mode inference.
    device : str
        The actual device string resolved.

    Raises
    ------
    ImportError
        If torch or terratorch are not installed.
    RuntimeError
        If model loading fails after downloading weights.
    """
    LightningInferenceModel = _require_terratorch()
    torch = _require_torch()

    resolved_device = resolve_device(device)
    logger.info("Loading model %s on device %s", name, resolved_device)

    config_path, weights_path = _download_weights(name)
    lightning_model = LightningInferenceModel.from_config(
        str(config_path), str(weights_path)
    )
    lightning_model.model.eval()
    lightning_model.model.to(resolved_device)

    logger.info("Model loaded successfully: %s", name)
    return lightning_model, resolved_device


def _make_hann_weight(tile: int) -> np.ndarray:
    """Create a 2-D Hanning (cosine taper) blending window of shape (tile, tile).

    Pixels at tile edges get weight ~0, pixels at centre get weight ~1.
    This ensures seamless blending when overlapping tiles.
    """
    hann_1d = np.hanning(tile).astype(np.float32)
    return np.outer(hann_1d, hann_1d)


def predict_water(
    s2_array: np.ndarray,
    tile: int = config.ML_TILE,
    overlap: int = config.ML_OVERLAP,
    batch: int = 1,
    lightning_model=None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Prithvi flood segmentation on a 6-band S2 array with tiled inference.

    Preprocessing (exact from config.yaml):
      1. Replace -9999 nodata pixels with 0.0.
      2. Multiply by constant_scale = 0.0001 (raw DN → reflectance).
    Tiling:
      Extract tiles of size (tile × tile) with stride = (tile - overlap).
      For each tile, run model forward pass, collect class-1 (water) probability.
      Blend tiles with a Hanning weight window; sum weights in an accumulator.
    Output:
      mask : uint8, 1=water, 0=non-water, 255=invalid (original nodata in any band).
      prob  : float32, water probability (0.0-1.0), nodata → 0.0.

    Parameters
    ----------
    s2_array : np.ndarray
        Shape (6, H, W), float32, raw DN scale (0-10000), nodata = -9999.
    tile : int
        Tile size in pixels.
    overlap : int
        Overlap in pixels between adjacent tiles.
    batch : int
        Number of tiles per forward pass (reduce for low-memory GPU).
    lightning_model :
        Loaded LightningInferenceModel instance.
    device : str
        Device string.

    Returns
    -------
    mask : np.ndarray, uint8, shape (H, W)
    prob  : np.ndarray, float32, shape (H, W)
    """
    torch = _require_torch()
    C, H, W = s2_array.shape
    if C != 6:
        raise ValueError(f"Expected 6-band S2 array, got shape {s2_array.shape}")

    # ── Build invalid mask (nodata in ANY band) ───────────────────────────────
    invalid = np.any(s2_array == NODATA_IN, axis=0)  # (H, W), True where nodata

    # ── Preprocess ────────────────────────────────────────────────────────────
    arr = s2_array.astype(np.float32).copy()
    arr = np.where(arr == NODATA_IN, NODATA_REPLACE, arr)
    arr = arr * CONSTANT_SCALE  # 0-10000 → 0.0-1.0

    # ── Accumulator arrays for Hanning blend ──────────────────────────────────
    prob_acc = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)
    hann_win = _make_hann_weight(tile)

    stride = tile - overlap

    # Collect tile coordinates
    tile_coords = []
    row = 0
    while True:
        col = 0
        while True:
            r1 = min(row, H - tile) if H >= tile else 0
            c1 = min(col, W - tile) if W >= tile else 0
            r2 = r1 + tile if H >= tile else H
            c2 = c1 + tile if W >= tile else W
            tile_coords.append((r1, c1, r2, c2))
            if col + stride >= W or W < tile:
                break
            col += stride
        if row + stride >= H or H < tile:
            break
        row += stride

    # Deduplicate while preserving order
    seen = set()
    unique_coords = []
    for coord in tile_coords:
        if coord not in seen:
            seen.add(coord)
            unique_coords.append(coord)

    model = lightning_model.model

    # ── Tile loop ─────────────────────────────────────────────────────────────
    batch_tiles = []
    batch_coords_list = []

    def _flush_batch(b_tiles, b_coords):
        """Run a batch of tiles through the model and accumulate results."""
        tensor_in = torch.tensor(np.stack(b_tiles, axis=0), device=device)
        # Shape: (B, 6, tile, tile) — model expects (B, C, H, W)
        with torch.no_grad():
            out = model(tensor_in)
            logits = out.output if hasattr(out, "output") else out
            # logits shape: (B, num_classes, H, W)
            probs = torch.softmax(logits.float(), dim=1)[:, 1, :, :].cpu().numpy()
        del tensor_in, logits
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        for prob_tile, (r1, c1, r2, c2) in zip(probs, b_coords):
            th, tw = r2 - r1, c2 - c1
            w = hann_win[:th, :tw]
            prob_acc[r1:r2, c1:c2] += prob_tile[:th, :tw] * w
            weight_acc[r1:r2, c1:c2] += w

    try:
        from tqdm import tqdm
        iterator = tqdm(unique_coords, desc="Prithvi inference tiles", unit="tile")
    except ImportError:
        iterator = unique_coords

    for coord in iterator:
        r1, c1, r2, c2 = coord
        tile_data = arr[:, r1:r2, c1:c2]  # (6, th, tw)

        # Pad smaller-than-tile patches (edge of image < tile)
        th, tw = r2 - r1, c2 - c1
        if th < tile or tw < tile:
            padded = np.zeros((6, tile, tile), dtype=np.float32)
            padded[:, :th, :tw] = tile_data
            tile_data = padded

        batch_tiles.append(tile_data)
        batch_coords_list.append(coord)

        if len(batch_tiles) == batch:
            _flush_batch(batch_tiles, batch_coords_list)
            batch_tiles = []
            batch_coords_list = []

    # Flush remainder
    if batch_tiles:
        _flush_batch(batch_tiles, batch_coords_list)

    # ── Normalise blended probabilities ───────────────────────────────────────
    weight_acc = np.where(weight_acc == 0, 1.0, weight_acc)  # avoid div-by-zero
    prob_map = (prob_acc / weight_acc).astype(np.float32)

    # ── Build mask ────────────────────────────────────────────────────────────
    mask = (prob_map >= config.ML_PROB_HIGH).astype(np.uint8)
    mask[invalid] = 255  # pass-through nodata as 255

    # Zero probability at nodata locations
    prob_map[invalid] = 0.0

    return mask, prob_map
