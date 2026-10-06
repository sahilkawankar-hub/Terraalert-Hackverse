"""
TerraAlert — Exposure join: zonal statistics and infrastructure overlay.

Computes per-zone flood metrics, population exposure, facility hits, and
road cut lengths.  All geometry operations use GRID_CRS (metric).

Inputs (from config paths):
    - flood raster (flood_fused.tif or flood_classical.tif)
    - outputs/pop.tif (population count per pixel)
    - data/roads.gpkg
    - data/facilities.gpkg

Outputs written:
    - outputs/roads_cut.geojson   (roads with in_flood boolean)
    - outputs/facilities.geojson  (facilities with in_flood boolean)

The function ``compute_exposure`` enriches the zone GeoDataFrame with:
    flood_km2, flood_pct, population, people_affected,
    facilities_hit, facilities_by_type, road_cut_km,
    confidence, confidence_score, reason

Zones below config.MIN_FLOOD_PCT are dropped.

CLI:
    python -m backend.zones.exposure_join   (not normally invoked directly)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("exposure_join")


# ---------------------------------------------------------------------------
# Flood area per zone
# ---------------------------------------------------------------------------

def _flood_stats(zones: gpd.GeoDataFrame, flood_path: Path) -> gpd.GeoDataFrame:
    """Add flood_km2 and flood_pct to zones (in GRID_CRS).

    Uses rasterstats to count flood pixels (value == 1) per zone, then
    converts to km² using the pixel size.
    """
    ps = config.PIXEL_SIZE  # metres

    with rasterio.open(flood_path) as src:
        affine = src.transform
        data = src.read(1)
        nodata = src.nodata
        src_crs = str(src.crs)

    # Ensure zones are in the raster CRS for zonal_stats
    if str(zones.crs) != src_crs:
        zones_reproj = zones.to_crs(src_crs)
    else:
        zones_reproj = zones

    stats = zonal_stats(
        zones_reproj,
        data,
        affine=affine,
        stats=["sum", "count"],
        nodata=nodata if nodata is not None else 255,
    )

    flood_pixels = np.array([s.get("sum", 0) or 0 for s in stats], dtype=float)
    total_pixels = np.array([s.get("count", 0) or 0 for s in stats], dtype=float)

    pixel_area_m2 = ps * ps
    zones = zones.copy()
    zones["flood_km2"] = (flood_pixels * pixel_area_m2) / 1_000_000.0
    zones["flood_pct"] = np.where(
        total_pixels > 0,
        (flood_pixels / total_pixels) * 100.0,
        0.0,
    )

    # Zone total area (from geometry)
    zones["zone_area_km2"] = zones.geometry.area / 1_000_000.0

    return zones


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _population_stats(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add population and people_affected to zones.

    Uses zonal_stats sum on pop.tif.  If pop.tif is missing, falls back to 0
    and records the fallback.

    people_affected = population * flood_pct / 100  (an estimate — the
    population raster counts everyone in the zone, not just those in the
    flooded area).
    """
    pop_path = config.OUTPUTS_DIR / "pop.tif"
    zones = zones.copy()

    if not pop_path.exists():
        logger.warning("pop.tif not found — population set to 0")
        zones["population"] = 0.0
        zones["people_affected"] = 0.0
        update_meta(
            config.OUTPUTS_DIR / "meta.json",
            "fallbacks.population",
            "pop.tif not found; population set to 0",
        )
        return zones

    with rasterio.open(pop_path) as src:
        affine = src.transform
        data = src.read(1)
        nodata = src.nodata
        src_crs = str(src.crs)

    if str(zones.crs) != src_crs:
        zones_reproj = zones.to_crs(src_crs)
    else:
        zones_reproj = zones

    stats = zonal_stats(
        zones_reproj,
        data,
        affine=affine,
        stats=["sum"],
        nodata=nodata if nodata is not None else -9999,
    )
    zones["population"] = [s.get("sum", 0) or 0 for s in stats]
    zones["people_affected"] = zones["population"] * zones["flood_pct"] / 100.0

    return zones


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------

def _facility_overlay(
    zones: gpd.GeoDataFrame,
    flood_polys: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Overlay facilities on flood polygons and aggregate per zone.

    Returns (zones_with_facility_cols, facilities_gdf_with_in_flood).
    """
    fac_path = config.DATA_DIR / "facilities.gpkg"
    zones = zones.copy()

    if not fac_path.exists():
        logger.warning("facilities.gpkg not found — facilities_hit = 0")
        zones["facilities_hit"] = 0
        zones["facilities_by_type"] = [{}] * len(zones)
        empty_fac = gpd.GeoDataFrame(
            columns=["osm_id", "facility_type", "name", "in_flood", "geometry"],
            crs="EPSG:4326",
        )
        return zones, empty_fac

    facilities = gpd.read_file(fac_path)
    if facilities.crs is None or str(facilities.crs) != config.GRID_CRS:
        facilities = facilities.to_crs(config.GRID_CRS)

    # Build flood union in GRID_CRS
    if len(flood_polys) > 0:
        flood_metric = flood_polys.to_crs(config.GRID_CRS) if str(flood_polys.crs) != config.GRID_CRS else flood_polys
        flood_union = unary_union(flood_metric.geometry)
    else:
        flood_union = None

    # Mark each facility as in_flood
    if flood_union is not None:
        facilities["in_flood"] = facilities.geometry.within(flood_union)
    else:
        facilities["in_flood"] = False

    # Spatial join: which zone does each facility fall in?
    fac_in_zone = gpd.sjoin(facilities, zones[["zone_id", "geometry"]], how="left", predicate="within")

    # Aggregate per zone
    hit_counts = (
        fac_in_zone[fac_in_zone["in_flood"]]
        .groupby("zone_id")
        .size()
        .rename("facilities_hit")
    )

    # Facilities by type per zone
    by_type = (
        fac_in_zone[fac_in_zone["in_flood"]]
        .groupby("zone_id")["facility_type"]
        .apply(lambda s: dict(s.value_counts()))
        .rename("facilities_by_type")
    )

    zones = zones.merge(hit_counts, on="zone_id", how="left")
    zones["facilities_hit"] = zones["facilities_hit"].fillna(0).astype(int)

    zones = zones.merge(by_type, on="zone_id", how="left")
    zones["facilities_by_type"] = zones["facilities_by_type"].apply(
        lambda v: v if isinstance(v, dict) else {}
    )

    # Prepare output facilities GeoJSON (EPSG:4326)
    fac_out = facilities[["osm_id", "facility_type", "name", "in_flood", "geometry"]].copy()
    fac_out = fac_out.to_crs("EPSG:4326")

    return zones, fac_out


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------

def _road_overlay(
    zones: gpd.GeoDataFrame,
    flood_polys: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Intersect roads with flood polygons and compute cut-length per zone.

    Returns (zones_with_road_cut_km, roads_gdf_with_in_flood).
    """
    road_path = config.DATA_DIR / "roads.gpkg"
    zones = zones.copy()

    if not road_path.exists():
        logger.warning("roads.gpkg not found — road_cut_km = 0")
        zones["road_cut_km"] = 0.0
        empty_roads = gpd.GeoDataFrame(
            columns=["osm_id", "highway", "name", "in_flood", "geometry"],
            crs="EPSG:4326",
        )
        return zones, empty_roads

    roads = gpd.read_file(road_path)
    if roads.crs is None or str(roads.crs) != config.GRID_CRS:
        roads = roads.to_crs(config.GRID_CRS)

    # Build flood union in GRID_CRS
    if len(flood_polys) > 0:
        flood_metric = flood_polys.to_crs(config.GRID_CRS) if str(flood_polys.crs) != config.GRID_CRS else flood_polys
        flood_union = unary_union(flood_metric.geometry)
    else:
        flood_union = None

    # Mark each road as in_flood (intersects)
    if flood_union is not None:
        roads["in_flood"] = roads.geometry.intersects(flood_union)
    else:
        roads["in_flood"] = False

    # For flooded roads, compute intersection length within each zone
    flooded_roads = roads[roads["in_flood"]].copy()

    if len(flooded_roads) > 0:
        # Clip flooded roads to flood extent
        flooded_segments = gpd.clip(flooded_roads, flood_metric)

        if len(flooded_segments) > 0:
            # Spatial join with zones to attribute cut length per zone
            seg_in_zone = gpd.sjoin(flooded_segments, zones[["zone_id", "geometry"]], how="left", predicate="intersects")
            cut_km = (
                seg_in_zone
                .assign(seg_len_km=seg_in_zone.geometry.length / 1000.0)
                .groupby("zone_id")["seg_len_km"]
                .sum()
                .rename("road_cut_km")
            )
            zones = zones.merge(cut_km, on="zone_id", how="left")
        else:
            zones["road_cut_km"] = 0.0
    else:
        zones["road_cut_km"] = 0.0

    zones["road_cut_km"] = zones["road_cut_km"].fillna(0.0).round(3)

    # Prepare output roads GeoJSON (EPSG:4326)
    roads_out = roads[["osm_id", "highway", "name", "in_flood", "geometry"]].copy()
    roads_out = roads_out.to_crs("EPSG:4326")

    return zones, roads_out


# ---------------------------------------------------------------------------
# Confidence fallback
# ---------------------------------------------------------------------------

def _add_confidence(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Try to call Person A's add_zone_confidence; fall back gracefully.

    The documented fallback is: confidence="Medium", confidence_score=0.5,
    reason="confidence module not run yet".
    """
    zones = zones.copy()
    try:
        from backend.fusion.confidence import add_zone_confidence
        zones = add_zone_confidence(zones, outputs_dir=config.OUTPUTS_DIR)
        logger.info("Confidence added by backend.fusion.confidence")
    except (ImportError, FileNotFoundError, Exception) as exc:
        logger.warning(
            "Confidence module unavailable (%s). Using fallback.", exc
        )
        zones["confidence"] = "Medium"
        zones["confidence_score"] = 0.5
        zones["reason"] = "confidence module not run yet"
        update_meta(
            config.OUTPUTS_DIR / "meta.json",
            "fallbacks.confidence",
            "confidence module not run yet; using Medium/0.5 fallback",
        )
    return zones


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def compute_exposure(
    zones: gpd.GeoDataFrame,
    flood_path: Path,
) -> gpd.GeoDataFrame:
    """Enrich zone GeoDataFrame with all exposure and confidence columns.

    All geometry operations are in GRID_CRS (metric).

    Parameters
    ----------
    zones : GeoDataFrame in GRID_CRS with zone_id, name, geometry.
    flood_path : Path to the flood raster (uint8, 1 = flood).

    Returns
    -------
    GeoDataFrame in GRID_CRS with contract columns added, zones below
    MIN_FLOOD_PCT dropped.
    """
    from backend.zones.build_zones import vectorise_flood

    # Ensure zones are in GRID_CRS
    if str(zones.crs) != config.GRID_CRS:
        zones = zones.to_crs(config.GRID_CRS)

    # 1. Flood area stats (raster-based)
    zones = _flood_stats(zones, flood_path)

    # 2. Population
    zones = _population_stats(zones)

    # 3. Vectorise flood for infrastructure overlay
    flood_polys = vectorise_flood(flood_path)

    # 4. Facilities
    zones, fac_gdf = _facility_overlay(zones, flood_polys)

    # 5. Roads
    zones, roads_gdf = _road_overlay(zones, flood_polys)

    # 6. Confidence
    zones = _add_confidence(zones)

    # 7. Drop zones below MIN_FLOOD_PCT
    n_before = len(zones)
    zones = zones[zones["flood_pct"] >= config.MIN_FLOOD_PCT].copy()
    n_dropped = n_before - len(zones)
    if n_dropped > 0:
        logger.info("Dropped %d zones below %.1f%% flood", n_dropped, config.MIN_FLOOD_PCT)

    # 8. Write infrastructure geojsons
    fac_out_path = config.OUTPUTS_DIR / "facilities.geojson"
    roads_out_path = config.OUTPUTS_DIR / "roads_cut.geojson"

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fac_gdf.to_file(fac_out_path, driver="GeoJSON")
    roads_gdf.to_file(roads_out_path, driver="GeoJSON")
    logger.info("Wrote %s (%d features)", fac_out_path.name, len(fac_gdf))
    logger.info("Wrote %s (%d features)", roads_out_path.name, len(roads_gdf))

    # Round numeric columns for clean output
    for col in ["flood_km2", "flood_pct", "population", "people_affected",
                "road_cut_km", "confidence_score"]:
        if col in zones.columns:
            zones[col] = zones[col].round(4)

    return zones
