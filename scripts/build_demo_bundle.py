"""
TerraAlert — Demo bundle packager.

Copies only the lightweight geospatial vectors, metadata, raster overlays,
and generated incident report into outputs/demo/ for self-contained frontend
hosting and git submission (excluding large raw GeoTIFF rasters).

Copies:
  - outputs/zones.geojson
  - outputs/flood_mask.geojson
  - outputs/facilities.geojson
  - outputs/roads_cut.geojson
  - outputs/meta.json
  - outputs/report.html
  - outputs/overlays/bounds.json
  - outputs/overlays/*.png (pre, post, flood, confidence)

CLI:
    python scripts/build_demo_bundle.py [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger

logger = get_logger("demo_bundle")

DEMO_DIR = config.OUTPUTS_DIR / "demo"

# Target files to bundle (relative to outputs/)
BUNDLE_ITEMS = [
    "zones.geojson",
    "flood_mask.geojson",
    "facilities.geojson",
    "roads_cut.geojson",
    "meta.json",
    "report.html",
    "overlays/bounds.json",
    "overlays/pre.png",
    "overlays/post.png",
    "overlays/flood.png",
    "overlays/confidence.png",
]


def format_size(bytes_num: int) -> str:
    """Format bytes into human-readable KiB or MiB."""
    if bytes_num < 1024:
        return f"{bytes_num} B"
    elif bytes_num < 1024 * 1024:
        return f"{bytes_num / 1024:.1f} KB"
    else:
        return f"{bytes_num / (1024 * 1024):.2f} MB"


def build_demo_bundle(force: bool = False) -> dict[str, int]:
    """Copy essential demo files to outputs/demo/ and compute size."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "overlays").mkdir(parents=True, exist_ok=True)

    copied: dict[str, int] = {}
    missing: list[str] = []

    print("=" * 60)
    print("Building TerraAlert Demo Bundle in outputs/demo/")
    print("=" * 60)

    for item in BUNDLE_ITEMS:
        src_path = config.OUTPUTS_DIR / item
        dst_path = DEMO_DIR / item

        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            size = dst_path.stat().st_size
            copied[item] = size
            print(f"  [COPIED] {item:<28} {format_size(size):>10}")
        elif dst_path.exists():
            # Already exists in demo directory (e.g. mock demo)
            size = dst_path.stat().st_size
            copied[item] = size
            print(f"  [RETAIN] {item:<28} {format_size(size):>10}")
        else:
            missing.append(item)
            print(f"  [SKIP]   {item:<28} (not found in outputs/)")

    total_bytes = sum(copied.values())

    print("-" * 60)
    print(f"Total bundled files: {len(copied)}")
    print(f"Total bundle size:  {format_size(total_bytes)} ({total_bytes:,} bytes)")
    if missing:
        print(f"Omitted optional/pending items ({len(missing)}): {', '.join(missing)}")
    print("=" * 60)

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-contained demo bundle")
    parser.add_argument("--force", action="store_true", help="Overwrite existing demo files")
    args = parser.parse_args()

    build_demo_bundle(force=args.force)


if __name__ == "__main__":
    main()
