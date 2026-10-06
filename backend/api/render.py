"""
TerraAlert — Raster overlay renderer.

Renders pre, post, flood, and confidence rasters to transparent PNGs
in EPSG:4326 under outputs/overlays/ and writes bounds.json with the
four corner coordinates in the order MapLibre image sources expect:
[[west, north], [east, north], [east, south], [west, south]].

Colormaps:
    pre/post  — grayscale (dB stretched linearly)
    flood     — semi-transparent blue
    confidence — red → yellow → green

CLI:
    python -m backend.api.render [--force]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("render")

OVERLAY_DIR = config.OUTPUTS_DIR / "overlays"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_raster_as_array(path: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Read a single-band raster, return (data, profile) or None if missing."""
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError:
        logger.error("rasterio required for overlay rendering")
        return None

    if not path.exists():
        return None

    with rasterio.open(path) as src:
        # Reproject to EPSG:4326 if needed
        if str(src.crs) != "EPSG:4326":
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.empty((height, width), dtype=src.dtypes[0])
            reproject(
                source=src.read(1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
            profile = {
                "transform": transform,
                "width": width,
                "height": height,
                "crs": "EPSG:4326",
                "nodata": src.nodata,
            }
        else:
            data = src.read(1)
            profile = {
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "crs": "EPSG:4326",
                "nodata": src.nodata,
            }

    return data, profile


def _bounds_from_profile(profile: dict[str, Any]) -> list[float]:
    """Extract [west, south, east, north] from a rasterio-like profile."""
    t = profile["transform"]
    w = profile["width"]
    h = profile["height"]
    west = t.c
    north = t.f
    east = t.c + t.a * w
    south = t.f + t.e * h
    return [west, south, east, north]


def _maplibre_bounds(bounds: list[float]) -> list[list[float]]:
    """Convert [west, south, east, north] to MapLibre corners order."""
    west, south, east, north = bounds
    return [
        [west, north],   # top-left
        [east, north],   # top-right
        [east, south],   # bottom-right
        [west, south],   # bottom-left
    ]


# ---------------------------------------------------------------------------
# Per-layer renderers
# ---------------------------------------------------------------------------

def _render_grayscale(data: np.ndarray, nodata: float | None, alpha: int = 200) -> Image.Image:
    """Render a float32 raster (e.g. dB backscatter) as grayscale PNG with alpha."""
    mask = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        mask = data != nodata

    valid = data[mask]
    if valid.size == 0:
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
        return Image.fromarray(rgba, mode="RGBA")

    vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    stretched = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    gray = (stretched * 255).astype(np.uint8)

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


def _render_flood(data: np.ndarray, nodata: float | None) -> Image.Image:
    """Render a uint8 flood mask (1 = flood) as semi-transparent blue."""
    flood = data == 1
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[flood, 0] = 30      # R
    rgba[flood, 1] = 100     # G
    rgba[flood, 2] = 220     # B
    rgba[flood, 3] = 160     # alpha
    return Image.fromarray(rgba, mode="RGBA")


def _render_confidence(data: np.ndarray, nodata: float | None) -> Image.Image:
    """Render a float32 confidence raster (0-1) as red → yellow → green."""
    mask = np.ones(data.shape, dtype=bool)
    if nodata is not None:
        mask = data != nodata
    mask &= np.isfinite(data)

    clamped = np.clip(data, 0, 1)
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)

    # Red to green via yellow: R = 255*(1-v), G = 255*v
    rgba[..., 0] = (255 * (1 - clamped)).astype(np.uint8)
    rgba[..., 1] = (255 * clamped).astype(np.uint8)
    rgba[..., 2] = 30
    rgba[..., 3] = np.where(mask, 160, 0).astype(np.uint8)

    return Image.fromarray(rgba, mode="RGBA")


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

LAYERS = {
    "pre": ("pre.tif", "grayscale"),
    "post": ("post.tif", "grayscale"),
    "flood": ("flood_classical.tif", "flood"),  # prefer fused if available
    "confidence": ("confidence_pixel.tif", "confidence"),
}


def render_overlays(force: bool = False) -> dict[str, Any]:
    """Render all available raster layers to PNGs and write bounds.json.

    Returns dict with rendered layer names and skipped reasons.
    """
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer fused flood mask if available
    flood_fused = config.OUTPUTS_DIR / "flood_fused.tif"
    flood_classical = config.OUTPUTS_DIR / "flood_classical.tif"
    if flood_fused.exists():
        LAYERS["flood"] = ("flood_fused.tif", "flood")
    elif flood_classical.exists():
        LAYERS["flood"] = ("flood_classical.tif", "flood")

    rendered: list[str] = []
    skipped: dict[str, str] = {}
    bounds_4326: list[float] | None = None

    for layer_name, (filename, colormap) in LAYERS.items():
        out_png = OVERLAY_DIR / f"{layer_name}.png"

        if out_png.exists() and not force:
            rendered.append(layer_name)
            logger.info("Skipping %s (exists). Use --force to regenerate.", layer_name)
            continue

        raster_path = config.OUTPUTS_DIR / filename
        result = _read_raster_as_array(raster_path)
        if result is None:
            reason = f"{filename} not found"
            skipped[layer_name] = reason
            logger.warning("Skipping %s overlay: %s", layer_name, reason)
            continue

        data, profile = result

        if colormap == "grayscale":
            img = _render_grayscale(data, profile.get("nodata"))
        elif colormap == "flood":
            img = _render_flood(data, profile.get("nodata"))
        elif colormap == "confidence":
            img = _render_confidence(data, profile.get("nodata"))
        else:
            skipped[layer_name] = f"unknown colormap: {colormap}"
            continue

        img.save(out_png)
        rendered.append(layer_name)
        logger.info("Rendered %s -> %s (%dx%d)", layer_name, out_png.name, img.width, img.height)

        # Use first valid layer for bounds
        if bounds_4326 is None:
            bounds_4326 = _bounds_from_profile(profile)

    # Fall back to config AOI if no raster was rendered
    if bounds_4326 is None:
        bounds_4326 = list(config.AOI_BBOX)

    # Write bounds.json for MapLibre
    bounds_data = {
        "bounds": bounds_4326,
        "corners": _maplibre_bounds(bounds_4326),
    }
    bounds_path = OVERLAY_DIR / "bounds.json"
    with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(bounds_data, f, indent=2)
        f.write("\n")
    logger.info("Wrote %s", bounds_path.name)

    update_meta(
        config.OUTPUTS_DIR / "meta.json",
        "overlays",
        {
            "rendered": rendered,
            "skipped": skipped,
        },
    )

    return {"rendered": rendered, "skipped": skipped, "bounds": bounds_4326}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Render raster overlays to PNGs")
    parser.add_argument("--force", action="store_true", help="Re-render even if PNGs exist")
    args = parser.parse_args()

    result = render_overlays(force=args.force)
    print(f"\n[DONE] Rendered: {result['rendered']}")
    if result["skipped"]:
        print(f"  Skipped: {result['skipped']}")


if __name__ == "__main__":
    main()
