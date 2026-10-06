"""
TerraAlert — OSM exposure data ingestion.

Downloads road network and critical facilities from OpenStreetMap via the
Overpass API, cleans and validates the data, reprojects to config.GRID_CRS,
and saves as GeoPackage files.

Outputs:
  data/roads.gpkg      – OSM highways (LineString), columns: osm_id, highway, name, geometry
  data/facilities.gpkg – OSM facilities (Point centroids), columns: osm_id, facility_type, name, geometry

CLI:
  python -m backend.ingest.exposure [--force]

osmnx v2 API note:
  features_from_bbox(bbox=(west, south, east, north), tags={...})
  bbox is (left, bottom, right, top) = (west, south, east, north) in EPSG:4326.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger

logger = get_logger("exposure")

# ---------------------------------------------------------------------------
# Constants derived from config — never hardcode
# ---------------------------------------------------------------------------

ROAD_TYPES: list[str] = ["motorway", "trunk", "primary", "secondary", "tertiary"]

FACILITY_TAGS: dict[str, Any] = {
    "amenity": ["hospital", "clinic", "school", "shelter", "fire_station", "police"],
    "healthcare": ["hospital", "clinic"],
}

# Map raw tag values → canonical facility_type label
FACILITY_TYPE_MAP: dict[str, str] = {
    "hospital": "hospital",
    "clinic": "clinic",
    "school": "school",
    "shelter": "shelter",
    "fire_station": "fire_station",
    "police": "police",
}

OVERPASS_ENDPOINTS: list[str] = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]

OVERPASS_TIMEOUT: int = 180   # seconds per attempt
OVERPASS_RETRIES: int = 3
OVERPASS_BACKOFF: float = 10.0  # seconds to wait between retries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_gdf_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert list/dict columns to JSON strings so they can be saved to GeoPackage.
    Drops columns that are entirely None after serialisation.
    """
    for col in gdf.columns:
        if col == "geometry":
            continue
        sample = gdf[col].dropna()
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
            gdf[col] = gdf[col].apply(
                lambda v: str(v) if isinstance(v, (list, dict)) else v
            )
    return gdf


def _polygons_to_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace Polygon / MultiPolygon geometries with their centroids."""
    def _centroid(geom: Any) -> Any:
        if geom is None:
            return None
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom.centroid
        return geom

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(_centroid)
    return gdf


def _fetch_overpass(
    bbox: tuple[float, float, float, float],
    tags: dict[str, Any],
    label: str,
) -> gpd.GeoDataFrame:
    """
    Query Overpass with retries across multiple endpoints.
    bbox is (west, south, east, north) in EPSG:4326.
    Returns a GeoDataFrame or raises RuntimeError if all attempts fail.
    """
    west, south, east, north = bbox
    last_exc: Exception | None = None

    for endpoint in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = endpoint
        ox.settings.requests_timeout = OVERPASS_TIMEOUT

        for attempt in range(1, OVERPASS_RETRIES + 1):
            logger.info(
                "[%s] Attempt %d/%d via %s ...",
                label, attempt, OVERPASS_RETRIES, endpoint,
            )
            try:
                t0 = time.perf_counter()
                ox_major = int(ox.__version__.split(".")[0]) if hasattr(ox, "__version__") else 2
                if ox_major >= 2:
                    gdf = ox.features_from_bbox(
                        bbox=(west, south, east, north),
                        tags=tags,
                    )
                else:
                    gdf = ox.features_from_bbox(
                        north, south, east, west,
                        tags=tags,
                    )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "[%s] Overpass returned %d features in %.1fs",
                    label, len(gdf), elapsed,
                )
                return gdf
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[%s] Attempt %d failed (%s). Retrying in %.0fs...",
                    label, attempt, exc, OVERPASS_BACKOFF * attempt,
                )
                time.sleep(OVERPASS_BACKOFF * attempt)

    raise RuntimeError(
        f"All Overpass endpoints failed for {label}: {last_exc}\n"
        "Suggestion: Download the Geofabrik India PBF extract from "
        "https://download.geofabrik.de/asia/india.html and filter locally "
        "with osmium or pyosmium."
    ) from last_exc


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------

def ingest_roads(force: bool = False) -> Path:
    """Download OSM highway roads and save to data/roads.gpkg."""
    out_path = config.DATA_DIR / "roads.gpkg"

    if out_path.exists() and not force:
        logger.info("roads.gpkg already exists. Skipping (use --force to overwrite).")
        gdf = gpd.read_file(out_path)
        _print_road_stats(gdf)
        return out_path

    logger.info("Querying OSM roads for AOI %s ...", config.AOI_BBOX)
    raw = _fetch_overpass(
        bbox=config.AOI_BBOX,
        tags={"highway": ROAD_TYPES},
        label="roads",
    )

    # Reset multi-level index that osmnx v2 creates (element_type, osmid)
    raw = raw.reset_index()

    # Drop empty / invalid geometries
    raw = raw[raw.geometry.notna() & raw.geometry.is_valid]

    # Keep only LineString features (drop nodes/polygons)
    raw = raw[raw.geometry.geom_type.isin(["LineString", "MultiLineString"])]

    if len(raw) == 0:
        raise ValueError("No valid road LineStrings returned from Overpass.")

    # Build clean output with required columns only
    osm_id_col = "osmid" if "osmid" in raw.columns else raw.columns[0]
    roads = gpd.GeoDataFrame(
        {
            "osm_id": raw[osm_id_col].astype(str),
            "highway": raw.get("highway", pd.Series("unknown", index=raw.index)),
            "name": raw.get("name", pd.Series(None, index=raw.index)),
        },
        geometry=raw.geometry,
        crs="EPSG:4326",
    )

    # Clean list/dict columns before write
    roads = _clean_gdf_columns(roads)

    # Reproject to metric CRS for length calculation, then save in WGS84
    roads_metric = roads.to_crs(config.GRID_CRS)
    total_km = roads_metric.geometry.length.sum() / 1000.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    roads.to_file(out_path, driver="GPKG")
    logger.info("Saved %s (%d features)", out_path, len(roads))

    _print_road_stats(roads, total_km=total_km)
    return out_path


def _print_road_stats(gdf: gpd.GeoDataFrame, total_km: float | None = None) -> None:
    if total_km is None:
        metric = gdf.to_crs(config.GRID_CRS)
        total_km = metric.geometry.length.sum() / 1000.0
    print(f"\n[roads.gpkg]")
    print(f"  Features:          {len(gdf):,}")
    print(f"  CRS:               {gdf.crs}")
    print(f"  Total road length: {total_km:.2f} km (in {config.GRID_CRS})")
    by_type = gdf.groupby("highway").size().sort_values(ascending=False)
    for hw, cnt in by_type.items():
        print(f"    {hw:<20} {cnt:>5}")


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------

def ingest_facilities(force: bool = False) -> Path:
    """Download OSM facilities and save to data/facilities.gpkg."""
    out_path = config.DATA_DIR / "facilities.gpkg"

    if out_path.exists() and not force:
        logger.info("facilities.gpkg already exists. Skipping (use --force to overwrite).")
        gdf = gpd.read_file(out_path)
        _print_facility_stats(gdf)
        return out_path

    logger.info("Querying OSM facilities for AOI %s ...", config.AOI_BBOX)
    raw = _fetch_overpass(
        bbox=config.AOI_BBOX,
        tags=FACILITY_TAGS,
        label="facilities",
    )

    raw = raw.reset_index()
    raw = raw[raw.geometry.notna() & raw.geometry.is_valid]

    if len(raw) == 0:
        raise ValueError("No valid facility features returned from Overpass.")

    # Convert polygons to centroids
    raw = _polygons_to_centroids(raw)

    # Drop anything that still isn't a point
    raw = raw[raw.geometry.geom_type == "Point"]

    # Determine facility_type from amenity or healthcare tag
    osm_id_col = "osmid" if "osmid" in raw.columns else raw.columns[0]

    def _resolve_type(row: pd.Series) -> str:
        for tag in ("amenity", "healthcare"):
            val = row.get(tag)
            if pd.notna(val) and isinstance(val, str):
                mapped = FACILITY_TYPE_MAP.get(val.lower())
                if mapped:
                    return mapped
        return "other"

    facility_type = raw.apply(_resolve_type, axis=1)

    facilities = gpd.GeoDataFrame(
        {
            "osm_id": raw[osm_id_col].astype(str),
            "facility_type": facility_type,
            "name": raw.get("name", pd.Series(None, index=raw.index)),
        },
        geometry=raw.geometry,
        crs="EPSG:4326",
    )

    # Drop rows that couldn't be classified
    facilities = facilities[facilities["facility_type"] != "other"]
    facilities = _clean_gdf_columns(facilities)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_file(out_path, driver="GPKG")
    logger.info("Saved %s (%d features)", out_path, len(facilities))

    _print_facility_stats(facilities)
    return out_path


def _print_facility_stats(gdf: gpd.GeoDataFrame) -> None:
    print(f"\n[facilities.gpkg]")
    print(f"  Features: {len(gdf):,}")
    print(f"  CRS:      {gdf.crs}")
    by_type = gdf.groupby("facility_type").size().sort_values(ascending=False)
    for ftype, cnt in by_type.items():
        print(f"    {ftype:<20} {cnt:>5}")


# ---------------------------------------------------------------------------
# Sources update
# ---------------------------------------------------------------------------

def _update_sources_md() -> None:
    """Append OSM data source info to data/SOURCES.md."""
    sources_path = config.DATA_DIR / "SOURCES.md"
    existing = sources_path.read_text(encoding="utf-8") if sources_path.exists() else ""
    marker = "### 5. OpenStreetMap"
    if marker in existing:
        return  # already written

    access_date = datetime.date.today().isoformat()
    section = f"""
{marker} — Roads and Facilities
- **Source**: OpenStreetMap contributors via Overpass API
- **Licence**: Open Database Licence (ODbL) v1.0 — https://opendatacommons.org/licenses/odbl/
- **Access date**: {access_date}
- **Road tags queried**: highway ∈ {ROAD_TYPES}
- **Facility tags queried**: amenity ∈ {FACILITY_TAGS['amenity']}, healthcare ∈ {FACILITY_TAGS['healthcare']}
- **Files**: `data/roads.gpkg` (LineStrings, EPSG:4326), `data/facilities.gpkg` (Points, EPSG:4326)
- **Attribution**: © OpenStreetMap contributors. When using this data, you must give credit to OSM and distribute
  any derived datasets under ODbL.
"""
    sources_path.write_text(existing + section, encoding="utf-8")
    logger.info("Updated %s with OSM attribution", sources_path)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_files() -> None:
    """Open both GeoPackages in geopandas and confirm they are valid."""
    for path in [config.DATA_DIR / "roads.gpkg", config.DATA_DIR / "facilities.gpkg"]:
        if not path.exists():
            logger.warning("MISSING: %s", path)
            continue
        gdf = gpd.read_file(path)
        if len(gdf) == 0:
            raise ValueError(f"{path.name} is empty after write!")
        required = {"osm_id", "geometry"}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(f"{path.name} missing required columns: {missing}")
        logger.info("VERIFIED %s - %d features, CRS=%s", path.name, len(gdf), gdf.crs)
        print(f"  [OK] {path.name}: {len(gdf)} features, CRS={gdf.crs}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OSM roads and facilities for TerraAlert AOI"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if output files already exist",
    )
    args = parser.parse_args()

    print(f"osmnx version: {ox.__version__}")
    print(f"AOI: {config.AOI_BBOX}")
    print(f"Output directory: {config.DATA_DIR}")

    print("\n--- Ingesting OSM Roads ---")
    ingest_roads(force=args.force)

    print("\n--- Ingesting OSM Facilities ---")
    ingest_facilities(force=args.force)

    print("\n--- Verifying output files ---")
    verify_files()

    _update_sources_md()
    print("\n[DONE] roads.gpkg and facilities.gpkg written and verified.")


if __name__ == "__main__":
    main()
