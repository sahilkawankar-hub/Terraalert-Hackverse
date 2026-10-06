"""
TerraAlert — Zone grid builder.

Creates a regular grid of ZONE_CELL_M-sized cells over the AOI in GRID_CRS,
vectorises the flood raster into outputs/flood_mask.geojson, then runs
exposure_join and priority scoring to produce outputs/zones.geojson with all
contract fields.

CLI:
    python -m backend.zones.build_zones [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio import features as rio_features
from shapely.geometry import box, mapping, shape

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("build_zones")


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid() -> gpd.GeoDataFrame:
    """Create a regular grid of ZONE_CELL_M cells over the AOI in GRID_CRS.

    Each cell is named ``Grid-<row>-<col>`` (0-indexed, row from top).

    Returns
    -------
    GeoDataFrame in GRID_CRS with columns: zone_id, name, geometry
    """
    west, south, east, north = config.AOI_BBOX
    cell = config.ZONE_CELL_M
    crs = config.GRID_CRS

    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_min, y_min = tx.transform(west, south)
    x_max, y_max = tx.transform(east, north)

    rows_data: list[dict[str, Any]] = []
    row_idx = 0
    y = y_max
    while y > y_min:
        col_idx = 0
        x = x_min
        while x < x_max:
            cell_box = box(x, max(y - cell, y_min), min(x + cell, x_max), y)
            zone_id = f"Grid-{row_idx}-{col_idx}"
            rows_data.append({
                "zone_id": zone_id,
                "name": zone_id,
                "geometry": cell_box,
            })
            col_idx += 1
            x += cell
        row_idx += 1
        y -= cell

    gdf = gpd.GeoDataFrame(rows_data, crs=crs)
    logger.info("Built %d grid zones (%d m cells)", len(gdf), cell)
    return gdf


# ---------------------------------------------------------------------------
# Flood mask vectorisation
# ---------------------------------------------------------------------------

def _resolve_flood_raster() -> Path:
    """Return the best available flood raster path.

    Prefers flood_fused.tif > flood_classical.tif.
    """
    fused = config.OUTPUTS_DIR / "flood_fused.tif"
    classical = config.OUTPUTS_DIR / "flood_classical.tif"
    if fused.exists():
        return fused
    if classical.exists():
        return classical
    raise FileNotFoundError(
        "No flood raster found.  Expected outputs/flood_fused.tif or "
        "outputs/flood_classical.tif.  Person A must produce these first."
    )


def vectorise_flood(flood_path: Path, out_path: Path | None = None) -> gpd.GeoDataFrame:
    """Convert a uint8 flood raster (1 = flood) to simplified polygons.

    Writes ``outputs/flood_mask.geojson`` in EPSG:4326.

    Returns
    -------
    GeoDataFrame of flood polygons in EPSG:4326.
    """
    if out_path is None:
        out_path = config.OUTPUTS_DIR / "flood_mask.geojson"

    with rasterio.open(flood_path) as src:
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs

    # Only polygonize flood pixels (value == 1) to avoid extracting non-flood areas
    flood_mask = (data == 1)
    if not np.any(flood_mask):
        logger.warning("No flood pixels found in %s", flood_path.name)
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON")
        return gdf

    flood_binary = flood_mask.astype(np.uint8)

    # Extract shapes only for flooded pixels
    polys = []
    for geom_dict, value in rio_features.shapes(flood_binary, mask=flood_mask, transform=transform):
        if value == 1:
            polys.append(shape(geom_dict))

    if not polys:
        logger.warning("No flood polygons extracted from %s", flood_path.name)
        gdf = gpd.GeoDataFrame(columns=["geometry"], crs=str(src_crs))
    else:
        gdf = gpd.GeoDataFrame(geometry=polys, crs=str(src_crs))

    # Reproject to 4326 if needed
    if gdf.crs and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Simplify for web display (~20 m tolerance in degrees at this latitude)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.0002, preserve_topology=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info("Wrote %s (%d flood polygons)", out_path.name, len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Full pipeline: grid → exposure → score → write
# ---------------------------------------------------------------------------

def run(force: bool = False) -> Path:
    """Build zones, compute exposure, score, and write outputs/zones.geojson.

    Returns path to zones.geojson.
    """
    zones_path = config.OUTPUTS_DIR / "zones.geojson"
    if zones_path.exists() and not force:
        logger.info("zones.geojson exists. Use --force to rebuild.")
        return zones_path

    # 1. Build grid
    zones = build_grid()

    # 2. Vectorise flood mask
    try:
        flood_path = _resolve_flood_raster()
        flood_gdf = vectorise_flood(flood_path)
        logger.info("Flood raster: %s", flood_path.name)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise

    # 3. Exposure join
    from backend.zones.exposure_join import compute_exposure
    zones = compute_exposure(zones, flood_path)

    # 4. Priority scoring
    from backend.priority.score import rescore
    # Convert GeoDataFrame → GeoJSON FeatureCollection dict
    zones_4326 = zones.to_crs("EPSG:4326")
    fc = json.loads(zones_4326.to_json())
    scored_fc = rescore(fc, config.WEIGHTS)

    # 5. Write zones.geojson
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(zones_path, "w", encoding="utf-8") as f:
        json.dump(scored_fc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote %s (%d zones)", zones_path.name, len(scored_fc.get("features", [])))

    # 6. Update meta.json
    update_meta(
        config.OUTPUTS_DIR / "meta.json",
        "zones",
        {
            "zone_source": config.ZONE_SOURCE,
            "zone_cell_m": config.ZONE_CELL_M,
            "total_zones": len(scored_fc.get("features", [])),
            "flood_raster": flood_path.name,
        },
    )

    return zones_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build priority zones from flood raster + exposure data",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = parser.parse_args()

    path = run(force=args.force)
    print(f"\n[DONE] {path}")


if __name__ == "__main__":
    main()
