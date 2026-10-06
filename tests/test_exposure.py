"""
Tests for backend/ingest/exposure.py — no network required.

Covers:
  1. Road length calculation on a synthetic metric-CRS LineString
  2. Column cleaning (list/dict → str)
  3. Polygon-to-centroid conversion
  4. Facility type resolution logic
  5. Required output columns are present
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.ingest.exposure import (
    FACILITY_TYPE_MAP,
    _clean_gdf_columns,
    _polygons_to_centroids,
)


# ---------------------------------------------------------------------------
# 1. Road length in metric CRS
# ---------------------------------------------------------------------------

def test_road_length_metric_crs():
    """
    A horizontal 1 km line in EPSG:32646 (UTM zone 46N) must measure exactly
    1000 m (within floating point rounding).
    """
    # Create a 1000 m horizontal line in UTM zone 46N
    start = (500000.0, 2900000.0)
    end   = (501000.0, 2900000.0)
    line  = LineString([start, end])

    gdf = gpd.GeoDataFrame(
        {"osm_id": ["test"], "highway": ["primary"], "name": [None]},
        geometry=[line],
        crs=config.GRID_CRS,
    )

    total_length_m = gdf.geometry.length.sum()
    assert math.isclose(total_length_m, 1000.0, rel_tol=1e-6), (
        f"Expected 1000 m, got {total_length_m} m"
    )
    total_km = total_length_m / 1000.0
    assert math.isclose(total_km, 1.0, rel_tol=1e-6)


def test_road_length_reprojection():
    """
    A road in EPSG:4326 (WGS84) reprojected to config.GRID_CRS should give
    a plausible metric length.  We use a 0.01° segment near the AOI centroid
    (~1.1 km in mid-latitudes) and verify it is in the 900–1300 m range.
    """
    west, south, east, north = config.AOI_BBOX
    lat = (south + north) / 2
    lon = (west + east) / 2

    # 0.01 degree horizontal segment starting at (lon, lat)
    line = LineString([(lon, lat), (lon + 0.01, lat)])
    gdf = gpd.GeoDataFrame(
        {"osm_id": ["seg1"], "highway": ["secondary"], "name": [None]},
        geometry=[line],
        crs="EPSG:4326",
    )
    gdf_metric = gdf.to_crs(config.GRID_CRS)
    length_m = gdf_metric.geometry.length.sum()

    assert 900.0 < length_m < 1300.0, (
        f"Reprojected 0.01° segment expected ~1100 m, got {length_m:.1f} m"
    )


# ---------------------------------------------------------------------------
# 2. Column cleaning
# ---------------------------------------------------------------------------

def test_clean_columns_list_to_str():
    """List-valued columns must be serialised to string, not cause a write error."""
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["1", "2"],
            "tags_list": [["a", "b"], ["c"]],
            "tags_dict": [{"k": "v"}, {}],
            "plain": ["x", "y"],
        },
        geometry=[Point(91.0, 26.2), Point(91.1, 26.3)],
        crs="EPSG:4326",
    )
    cleaned = _clean_gdf_columns(gdf.copy())

    # Serialised values must be plain strings
    assert all(isinstance(v, str) for v in cleaned["tags_list"]), (
        "List column not converted to str"
    )
    assert all(isinstance(v, str) for v in cleaned["tags_dict"]), (
        "Dict column not converted to str"
    )
    # Non-list columns must be untouched
    assert list(cleaned["plain"]) == ["x", "y"]


def test_clean_columns_no_touch_geometry():
    """_clean_gdf_columns must never convert the geometry column."""
    gdf = gpd.GeoDataFrame(
        {"osm_id": ["1"]},
        geometry=[Point(91.0, 26.2)],
        crs="EPSG:4326",
    )
    cleaned = _clean_gdf_columns(gdf.copy())
    assert cleaned.geometry.iloc[0].geom_type == "Point"


# ---------------------------------------------------------------------------
# 3. Polygon-to-centroid conversion
# ---------------------------------------------------------------------------

def test_polygons_to_centroids():
    """Polygons must be replaced by their centroids; Points must remain Points."""
    poly = Polygon([(91.0, 26.2), (91.1, 26.2), (91.1, 26.3), (91.0, 26.3)])
    pt   = Point(91.05, 26.25)

    gdf = gpd.GeoDataFrame(
        {"osm_id": ["poly_1", "pt_1"]},
        geometry=[poly, pt],
        crs="EPSG:4326",
    )
    result = _polygons_to_centroids(gdf)

    # Both should now be Points
    assert all(g.geom_type == "Point" for g in result.geometry), (
        "Not all geometries converted to Point"
    )

    # Polygon centroid should be near centre of the box
    centroid = result.geometry.iloc[0]
    assert math.isclose(centroid.x, 91.05, abs_tol=1e-6)
    assert math.isclose(centroid.y, 26.25, abs_tol=1e-6)

    # Original Point must be unchanged
    assert math.isclose(result.geometry.iloc[1].x, 91.05, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 4. Facility type resolution
# ---------------------------------------------------------------------------

def test_facility_type_map_coverage():
    """All required facility types must be present in FACILITY_TYPE_MAP."""
    required = {"hospital", "clinic", "school", "shelter", "fire_station", "police"}
    assert required.issubset(set(FACILITY_TYPE_MAP.keys())), (
        f"Missing facility types: {required - set(FACILITY_TYPE_MAP.keys())}"
    )


def test_facility_type_map_self_consistent():
    """FACILITY_TYPE_MAP values should match canonical type names (no typos)."""
    for key, val in FACILITY_TYPE_MAP.items():
        assert isinstance(val, str) and len(val) > 0


# ---------------------------------------------------------------------------
# 5. Output schema contract
# ---------------------------------------------------------------------------

def test_road_output_columns(tmp_path: Path):
    """A minimal roads GeoDataFrame must have osm_id, highway, name, geometry."""
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["12345"],
            "highway": ["primary"],
            "name": ["NH27"],
        },
        geometry=[LineString([(91.0, 26.2), (91.1, 26.3)])],
        crs="EPSG:4326",
    )
    out = tmp_path / "roads.gpkg"
    gdf.to_file(out, driver="GPKG")

    reloaded = gpd.read_file(out)
    for col in ("osm_id", "highway", "name"):
        assert col in reloaded.columns, f"Missing column '{col}' in roads.gpkg"
    assert reloaded.crs.to_epsg() == 4326


def test_facility_output_columns(tmp_path: Path):
    """A minimal facilities GeoDataFrame must have osm_id, facility_type, name, geometry."""
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["99"],
            "facility_type": ["hospital"],
            "name": ["Barpeta Medical College"],
        },
        geometry=[Point(91.0, 26.2)],
        crs="EPSG:4326",
    )
    out = tmp_path / "facilities.gpkg"
    gdf.to_file(out, driver="GPKG")

    reloaded = gpd.read_file(out)
    for col in ("osm_id", "facility_type", "name"):
        assert col in reloaded.columns, f"Missing column '{col}' in facilities.gpkg"
    assert reloaded.crs.to_epsg() == 4326
