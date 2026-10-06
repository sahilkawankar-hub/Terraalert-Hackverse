"""
TerraAlert — centralised project configuration.

ALL tuneable constants live here. Other modules import from config; they never
hardcode AOI coordinates, dates, CRS, pixel sizes, thresholds, or weights.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
DATA_DIR: Path = ROOT_DIR / "data"
GEE_CACHE_DIR: Path = DATA_DIR / "gee_cache"

# ── Event ────────────────────────────────────────────────────────────────────
EVENT_NAME: str = "Assam Floods, June 2022 (Barpeta-Nalbari, Brahmaputra north bank)"

# ── Area of Interest (west, south, east, north — EPSG:4326) ─────────────────
AOI_BBOX: tuple[float, float, float, float] = (90.95, 26.10, 91.25, 26.37)

# ── Temporal windows ─────────────────────────────────────────────────────────
PRE_START: date = date(2022, 4, 20)
PRE_END: date = date(2022, 5, 10)
POST_START: date = date(2022, 6, 17)
POST_END: date = date(2022, 6, 30)

# ── Google Earth Engine ──────────────────────────────────────────────────────
GEE_PROJECT: str = os.environ.get("GEE_PROJECT", "terraalert-hackverse")

# ── Reference grid ───────────────────────────────────────────────────────────
GRID_CRS: str = "EPSG:32646"          # UTM zone 46N (covers Assam)
PIXEL_SIZE: int = 20                   # metres

# ── Sentinel-1 ───────────────────────────────────────────────────────────────
S1_PASS: str = "DESCENDING"
S1_RELATIVE_ORBIT: Optional[int] = 150
S1_BAND: str = "VV"

# ── Classical detection thresholds ───────────────────────────────────────────
POST_DB_MAX: float = -18.0            # post VV must be darker than this (water)
DIFF_DB_MAX: float = -3.0             # post – pre must drop by at least this
SLOPE_MAX_DEG: float = 5.0            # exclude steep terrain from flood mask
STEEP_FLAG_DEG: float = 15.0          # flag as "steep terrain" in confidence
MIN_OBJECT_PIXELS: int = 25           # minimum connected-component size
MIN_FLOOD_PCT: float = 1.0            # zone must have ≥ 1 % flood to rank

# ── Priority scoring weights ────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "severity": 0.30,
    "people": 0.30,
    "facilities": 0.20,
    "roads": 0.20,
}

TIER_P1_FRAC: float = 0.20            # top 20 % → P1
TIER_P2_FRAC: float = 0.30            # next 30 % → P2, rest → P3

# ── Confidence scoring ──────────────────────────────────────────────────────
CONF_WEIGHTS: dict[str, float] = {
    "agreement": 0.5,
    "terrain": 0.3,
    "time": 0.2,
}
CONF_HIGH: float = 0.70
CONF_LOW: float = 0.40

# ── Zones ────────────────────────────────────────────────────────────────────
ZONE_SOURCE: str = "grid"              # "grid" | future: "admin"
ZONE_CELL_M: int = 1000               # grid cell size in metres

# ── Demo mode ────────────────────────────────────────────────────────────────
DEMO_MODE: bool = os.environ.get("DEMO_MODE", "0") == "1"

# ── ML / Prithvi model ──────────────────────────────────────────────────────
ML_MODEL: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
ML_MODEL_FALLBACK: str = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11"
ML_DEVICE: str = "auto"
ML_TILE: int = 512
ML_OVERLAP: int = 64
ML_PROB_HIGH: float = 0.8
FUSION_RULE: str = "classical_plus_ml"

# --- A settings ---
USE_OTSU: bool = False
ML_BATCH_SIZE: int = 1          # tiles per forward pass; reduce to 1 if GPU OOM
S2_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"   # confirmed active collection ID

# Updated confidence scoring weights for multi-method fusion:
# Formula:
#   base_score = (CONF_WEIGHTS['agreement'] * agreement_ratio)
#              + (CONF_WEIGHTS['margin'] * mean_margin)
#              + (CONF_WEIGHTS['terrain'] * (1.0 - terrain_penalty))
#              + (CONF_WEIGHTS['time'] * (1.0 - time_penalty))
#   final_score = clip(base_score - method_penalty, 0.0, 1.0)
# where:
#   agreement_ratio = share of flooded pixels in zone where both methods agree (agreement == 2)
#   method_penalty  = (share of zone with ML unavailable) * METHOD_PENALTY_MAX
CONF_WEIGHTS: dict[str, float] = {
    "agreement": 0.35,  # consensus between SAR classical and Optical ML
    "margin": 0.20,     # SAR detection margin certainty
    "terrain": 0.25,    # slope stability (1.0 - terrain_penalty)
    "time": 0.20,       # temporal stability (1.0 - time_penalty)
}
METHOD_PENALTY_MAX: float = 0.15  # penalty applied when ML is unavailable for a zone

# --- B settings ---

