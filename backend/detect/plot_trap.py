"""
TerraAlert — Disaster Trap Comparison visualization.

Plots 3 side-by-side panels:
  1. Naive single-image baseline (post < -18 dB) — traps permanent river.
  2. TerraAlert classical change detection — excludes permanent river.
  3. Sentinel-1 post-event VV backscatter (dB) — radar reference.

Saves to outputs/trap_comparison.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import config
from backend.common.log import get_logger

logger = get_logger("detect.plot_trap")


def generate_trap_comparison(
    outputs_dir: Optional[Path] = None,
    out_png: Optional[Path] = None,
    crop_slice: tuple[slice, slice] = (slice(700, 1100), slice(500, 1100)),
) -> Path:
    """
    Generate 3-panel comparison figure demonstrating avoidance of the disaster trap.
    """
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR
    if out_png is None:
        out_png = outputs_dir / "trap_comparison.png"

    post_path = outputs_dir / "post.tif"
    naive_path = outputs_dir / "flood_naive.tif"
    classical_path = outputs_dir / "flood_classical.tif"
    perm_path = outputs_dir / "perm_water.tif"

    for p in (post_path, naive_path, classical_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing raster required for comparison plot: {p}")

    row_s, col_s = crop_slice

    with rasterio.open(post_path) as src:
        post = src.read(1)[row_s, col_s]
    with rasterio.open(naive_path) as src:
        naive = src.read(1)[row_s, col_s]
    with rasterio.open(classical_path) as src:
        classical = src.read(1)[row_s, col_s]
    with rasterio.open(perm_path) as src:
        perm = src.read(1)[row_s, col_s]

    # Create dark-mode / high-contrast figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=200)

    # Base post-image display range (-25 dB to -5 dB)
    post_clipped = np.clip(post, -25.0, -5.0)

    # 1. Panel 1: Naive Baseline
    ax1 = axes[0]
    ax1.imshow(post_clipped, cmap="gray", vmin=-25, vmax=-5)
    # Overlay naive flood in fiery orange/red
    naive_overlay = np.ma.masked_where(naive != 1, naive)
    ax1.imshow(naive_overlay, cmap=mcolors.ListedColormap(["#ff4d4d"]), alpha=0.85)
    ax1.set_title("Naive Single-Image (post < -18 dB)\n[TRAP: Flags Permanent River as Flood]", fontsize=12, color="#ff6b6b", fontweight="bold")
    ax1.axis("off")

    # 2. Panel 2: Classical Change Detection
    ax2 = axes[1]
    ax2.imshow(post_clipped, cmap="gray", vmin=-25, vmax=-5)
    # Overlay permanent river faintly in deep blue for context
    perm_overlay = np.ma.masked_where(perm != 1, perm)
    ax2.imshow(perm_overlay, cmap=mcolors.ListedColormap(["#1e3799"]), alpha=0.45)
    # Overlay classical flood in bright cyan
    classical_overlay = np.ma.masked_where(classical != 1, classical)
    ax2.imshow(classical_overlay, cmap=mcolors.ListedColormap(["#00d2d3"]), alpha=0.95)
    ax2.set_title("TerraAlert Classical Change Detection\n[CORRECT: Excludes River & Steep Terrain]", fontsize=12, color="#00d2d3", fontweight="bold")
    ax2.axis("off")

    # 3. Panel 3: Sentinel-1 Post VV Backscatter
    ax3 = axes[2]
    im = ax3.imshow(post_clipped, cmap="viridis", vmin=-25, vmax=-5)
    ax3.set_title("Sentinel-1 Post-Event Backscatter (VV dB)\n[Radar Observation Reference]", fontsize=12, color="#feca57", fontweight="bold")
    ax3.axis("off")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Backscatter (dB)", color="#feca57")

    fig.suptitle(
        f"The Disaster Trap Comparison — {config.EVENT_NAME}\n"
        "Multi-temporal differencing + JRC permanent water masking prevents massive false-alarm classification",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", facecolor="#121212")
    plt.close(fig)

    logger.info("Saved trap comparison figure to %s", out_png)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trap comparison figure")
    parser.parse_args()
    generate_trap_comparison()


if __name__ == "__main__":
    main()
