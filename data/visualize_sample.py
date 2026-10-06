#!/usr/bin/env python3
"""
Convert Sen1Floods11 multi-band GeoTIFF tiles into viewable standard PNG images:
- Sentinel-2 True Color (RGB: Bands 4, 3, 2)
- Sentinel-1 SAR Radar Composite (VV, VH, and Polarization Difference)
- Ground Truth Flood Mask
- JRC Permanent Water Reference Mask
- Side-by-Side Comparison Panels
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

def normalize_band(band, p_min=2, p_max=98):
    """Percentile normalization for satellite reflectance/radar dB."""
    p_low, p_high = np.percentile(band, (p_min, p_max))
    if p_high - p_low < 1e-6:
        return np.zeros_like(band, dtype=float)
    return np.clip((band - p_low) / (p_high - p_low), 0, 1)

def export_scene_visualization(scene_id: str, raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_path = raw_dir / "S1Hand" / f"{scene_id}_S1Hand.tif"
    s2_path = raw_dir / "S2Hand" / f"{scene_id}_S2Hand.tif"
    label_path = raw_dir / "LabelHand" / f"{scene_id}_LabelHand.tif"
    jrc_path = raw_dir / "JRCWaterHand" / f"{scene_id}_JRCWaterHand.tif"

    if not all(p.exists() for p in [s1_path, s2_path, label_path, jrc_path]):
        print(f"Skipping {scene_id}: one or more sensor files missing.")
        return None

    # 1. Sentinel-2 True-Color RGB (Band 4 = Red, Band 3 = Green, Band 2 = Blue)
    with rasterio.open(s2_path) as src:
        red = src.read(4).astype(float)
        green = src.read(3).astype(float)
        blue = src.read(2).astype(float)
        rgb = np.stack([normalize_band(red), normalize_band(green), normalize_band(blue)], axis=-1)

    # 2. Sentinel-1 SAR Radar Composite (Band 1 = VV, Band 2 = VH)
    with rasterio.open(s1_path) as src:
        vv = src.read(1).astype(float)
        vh = src.read(2).astype(float)
        ratio = vv - vh
        sar_rgb = np.stack([normalize_band(vv), normalize_band(vh), normalize_band(ratio)], axis=-1)

    # 3. Hand-annotated Flood Mask (-1: No data / Cloud, 0: Land, 1: Water/Flood)
    with rasterio.open(label_path) as src:
        label = src.read(1)

    # 4. JRC Permanent Water Mask (0: No water, 1: Seasonal water, 2: Permanent water)
    with rasterio.open(jrc_path) as src:
        jrc = src.read(1)

    # Plot Comparison Grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 (Optical RGB)", fontsize=13, weight="bold")
    axes[0].axis("off")

    axes[1].imshow(sar_rgb)
    axes[1].set_title("Sentinel-1 SAR Radar (VV / VH)", fontsize=13, weight="bold")
    axes[1].axis("off")

    im_label = axes[2].imshow(label, cmap="Blues", vmin=-1, vmax=1)
    axes[2].set_title("Ground-Truth Flood Mask", fontsize=13, weight="bold")
    axes[2].axis("off")

    axes[3].imshow(jrc, cmap="viridis")
    axes[3].set_title("JRC Permanent Water Mask", fontsize=13, weight="bold")
    axes[3].axis("off")

    fig.suptitle(f"TerraAlert Data Preview — Scene: {scene_id}", fontsize=16, weight="bold")
    out_file = out_dir / f"{scene_id}_preview.png"
    plt.savefig(out_file, dpi=180)
    plt.close()

    print(f"Exported preview: {out_file}")
    return out_file

def main():
    parser = argparse.ArgumentParser(description="Export Sen1Floods11 GeoTIFF tiles to PNG previews")
    parser.add_argument("--count", type=int, default=3, help="Number of sample scenes to export")
    parser.add_argument("--scene", type=str, default=None, help="Specific scene ID (e.g. India_25540)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "previews"

    if args.scene:
        scenes = [args.scene]
    else:
        label_files = sorted((raw_dir / "LabelHand").glob("India_*_LabelHand.tif"))
        scenes = [f.stem.replace("_LabelHand", "") for f in label_files[:args.count]]

    for scene in scenes:
        export_scene_visualization(scene, raw_dir, out_dir)

if __name__ == "__main__":
    main()
