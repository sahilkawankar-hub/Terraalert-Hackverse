"""
Tests for backend/zones/ and backend/priority/score.py — synthetic only.

All fixtures create temporary rasters and vector files in tmp_path.
No flood rasters are ever written to outputs/.

Covers:
  1. Grid construction (correct cell count, naming)
  2. Flood raster vectorisation
  3. Flood stats (flood_km2, flood_pct)
  4. Population stats (people_affected)
  5. Road cut length
  6. Facility hit count + by-type
  7. Priority scoring (ranking, tiers, VERIFY rule)
  8. Changing weights changes ranking
  9. Empty zone (no road, no facility) does not crash
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely.geometry import LineString, Point, box

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


# ============================================================================
# Helpers: create synthetic rasters and vector data
# ============================================================================

def _make_flood_raster(
    path: Path,
    flood_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write a small uint8 flood raster aligned to the reference grid.

    If flood_rect (x_min, y_min, x_max, y_max in GRID_CRS) is given,
    pixels inside it are set to 1 (flooded).
    """
    from grid import reference_grid
    g = reference_grid()

    data = np.zeros((g["height"], g["width"]), dtype=np.uint8)
    transform: Affine = g["transform"]

    if flood_rect is not None:
        fx_min, fy_min, fx_max, fy_max = flood_rect
        col_min = max(0, int((fx_min - transform.c) / transform.a))
        col_max = min(g["width"], int(math.ceil((fx_max - transform.c) / transform.a)))
        row_min = max(0, int((fy_max - transform.f) / transform.e))
        row_max = min(g["height"], int(math.ceil((fy_min - transform.f) / transform.e)))
        data[row_min:row_max, col_min:col_max] = 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=g["width"],
        height=g["height"],
        count=1,
        dtype="uint8",
        crs=g["crs"],
        transform=transform,
        nodata=255,
    ) as dst:
        dst.write(data, 1)

    return path


def _make_pop_raster(path: Path, pop_per_pixel: float = 10.0) -> Path:
    """Write a constant-value population raster aligned to the reference grid."""
    from grid import reference_grid
    g = reference_grid()

    data = np.full((g["height"], g["width"]), pop_per_pixel, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=g["width"],
        height=g["height"],
        count=1,
        dtype="float32",
        crs=g["crs"],
        transform=g["transform"],
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    return path


def _make_roads_gpkg(path: Path, lines: list[LineString] | None = None) -> Path:
    """Write a small roads GeoPackage in GRID_CRS."""
    if lines is None:
        lines = []
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": [str(i) for i in range(len(lines))],
            "highway": ["primary"] * len(lines),
            "name": [f"Road-{i}" for i in range(len(lines))],
        },
        geometry=lines,
        crs=config.GRID_CRS,
    )
    # Save in WGS84 (like the real data)
    gdf = gdf.to_crs("EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    return path


def _make_facilities_gpkg(
    path: Path,
    points: list[Point] | None = None,
    types: list[str] | None = None,
) -> Path:
    """Write a small facilities GeoPackage in GRID_CRS."""
    if points is None:
        points = []
    if types is None:
        types = ["hospital"] * len(points)
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": [str(i) for i in range(len(points))],
            "facility_type": types,
            "name": [f"Fac-{i}" for i in range(len(points))],
        },
        geometry=points,
        crs=config.GRID_CRS,
    )
    gdf = gdf.to_crs("EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    return path


def _get_aoi_metric_bounds() -> tuple[float, float, float, float]:
    """Get AOI bounds in GRID_CRS."""
    tx = Transformer.from_crs("EPSG:4326", config.GRID_CRS, always_xy=True)
    west, south, east, north = config.AOI_BBOX
    x_min, y_min = tx.transform(west, south)
    x_max, y_max = tx.transform(east, north)
    return x_min, y_min, x_max, y_max


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synth_env(tmp_path: Path, monkeypatch):
    """Set up synthetic flood raster + exposure data in tmp_path.

    Creates a flood rectangle covering one grid cell near the top-left corner,
    a road crossing through it, and a hospital inside it.
    """
    x_min, y_min, x_max, y_max = _get_aoi_metric_bounds()

    # Flood rectangle: first 1000m cell (top-left)
    cell = config.ZONE_CELL_M
    flood_rect = (x_min, y_max - cell, x_min + cell, y_max)

    # Create directories
    out_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()

    # Monkeypatch config paths
    monkeypatch.setattr(config, "OUTPUTS_DIR", out_dir)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)

    # Flood raster
    flood_path = out_dir / "flood_classical.tif"
    _make_flood_raster(flood_path, flood_rect)

    # Population raster (10 people per pixel)
    _make_pop_raster(out_dir / "pop.tif", pop_per_pixel=10.0)

    # Road: a horizontal 1km line through the flooded cell
    road_y = y_max - cell / 2
    road_line = LineString([
        (x_min, road_y),
        (x_min + cell, road_y),
    ])
    _make_roads_gpkg(data_dir / "roads.gpkg", [road_line])

    # Facility: a hospital inside the flooded cell
    hospital_pt = Point(x_min + cell / 2, y_max - cell / 2)
    _make_facilities_gpkg(
        data_dir / "facilities.gpkg",
        [hospital_pt],
        ["hospital"],
    )

    # Write empty meta.json
    (out_dir / "meta.json").write_text("{}", encoding="utf-8")

    return {
        "flood_path": flood_path,
        "flood_rect": flood_rect,
        "out_dir": out_dir,
        "data_dir": data_dir,
        "cell": cell,
    }


# ============================================================================
# Tests: Grid construction
# ============================================================================

class TestBuildGrid:

    def test_grid_covers_aoi(self):
        from backend.zones.build_zones import build_grid
        zones = build_grid()
        assert len(zones) > 0
        assert "zone_id" in zones.columns
        assert "name" in zones.columns

    def test_grid_naming(self):
        from backend.zones.build_zones import build_grid
        zones = build_grid()
        # First zone should be Grid-0-0
        assert zones.iloc[0]["zone_id"] == "Grid-0-0"

    def test_grid_cell_count(self):
        from backend.zones.build_zones import build_grid
        zones = build_grid()
        x_min, y_min, x_max, y_max = _get_aoi_metric_bounds()
        expected_cols = math.ceil((x_max - x_min) / config.ZONE_CELL_M)
        expected_rows = math.ceil((y_max - y_min) / config.ZONE_CELL_M)
        # Total zones should be rows * cols
        assert len(zones) == expected_rows * expected_cols

    def test_grid_crs(self):
        from backend.zones.build_zones import build_grid
        zones = build_grid()
        assert str(zones.crs) == config.GRID_CRS


# ============================================================================
# Tests: Flood vectorisation
# ============================================================================

class TestVectoriseFlood:

    def test_vectorise_produces_polygons(self, synth_env, tmp_path):
        from backend.zones.build_zones import vectorise_flood

        out_path = tmp_path / "test_flood_mask.geojson"
        gdf = vectorise_flood(synth_env["flood_path"], out_path)

        assert len(gdf) > 0
        assert out_path.exists()
        assert str(gdf.crs) == "EPSG:4326"

    def test_vectorise_empty_raster(self, tmp_path, monkeypatch):
        """A raster with no flood pixels should produce zero polygons."""
        monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
        from backend.zones.build_zones import vectorise_flood

        flood_path = tmp_path / "empty_flood.tif"
        _make_flood_raster(flood_path, flood_rect=None)

        out_path = tmp_path / "empty_flood_mask.geojson"
        gdf = vectorise_flood(flood_path, out_path)
        assert len(gdf) == 0


# ============================================================================
# Tests: Exposure join
# ============================================================================

class TestExposureJoin:

    def test_flood_stats(self, synth_env):
        from backend.zones.build_zones import build_grid
        from backend.zones.exposure_join import _flood_stats

        zones = build_grid()
        zones = _flood_stats(zones, synth_env["flood_path"])

        assert "flood_km2" in zones.columns
        assert "flood_pct" in zones.columns

        # At least one zone should have flood_pct > 0
        assert zones["flood_pct"].max() > 0

    def test_population_stats(self, synth_env):
        from backend.zones.build_zones import build_grid
        from backend.zones.exposure_join import _flood_stats, _population_stats

        zones = build_grid()
        zones = _flood_stats(zones, synth_env["flood_path"])
        zones = _population_stats(zones)

        assert "population" in zones.columns
        assert "people_affected" in zones.columns
        assert zones["population"].sum() > 0

        # people_affected = population * flood_pct / 100
        flooded = zones[zones["flood_pct"] > 0].iloc[0]
        expected_pa = flooded["population"] * flooded["flood_pct"] / 100.0
        assert math.isclose(flooded["people_affected"], expected_pa, rel_tol=0.01)

    def test_facility_hit(self, synth_env):
        from backend.zones.build_zones import build_grid, vectorise_flood
        from backend.zones.exposure_join import _flood_stats, _facility_overlay

        zones = build_grid()
        zones = _flood_stats(zones, synth_env["flood_path"])
        flood_polys = vectorise_flood(synth_env["flood_path"])

        zones, fac_gdf = _facility_overlay(zones, flood_polys)

        assert "facilities_hit" in zones.columns
        # The hospital is inside the flood, so at least 1 hit
        total_hit = zones["facilities_hit"].sum()
        assert total_hit >= 1, f"Expected ≥1 facility hit, got {total_hit}"

    def test_road_cut_km(self, synth_env):
        from backend.zones.build_zones import build_grid, vectorise_flood
        from backend.zones.exposure_join import _flood_stats, _road_overlay

        zones = build_grid()
        zones = _flood_stats(zones, synth_env["flood_path"])
        flood_polys = vectorise_flood(synth_env["flood_path"])

        zones, roads_gdf = _road_overlay(zones, flood_polys)

        assert "road_cut_km" in zones.columns
        total_cut = zones["road_cut_km"].sum()
        # Road is ~1 km through the flooded cell
        assert total_cut > 0, f"Expected road_cut_km > 0, got {total_cut}"

    def test_no_road_no_facility_no_crash(self, synth_env, monkeypatch):
        """Zone with missing road/facility files should not crash."""
        from backend.zones.build_zones import build_grid, vectorise_flood
        from backend.zones.exposure_join import _facility_overlay, _road_overlay

        # Remove data files
        (synth_env["data_dir"] / "roads.gpkg").unlink()
        (synth_env["data_dir"] / "facilities.gpkg").unlink()

        zones = build_grid()
        flood_polys = vectorise_flood(synth_env["flood_path"])

        zones, fac_gdf = _facility_overlay(zones, flood_polys)
        zones, roads_gdf = _road_overlay(zones, flood_polys)

        assert zones["facilities_hit"].sum() == 0
        assert zones["road_cut_km"].sum() == 0.0

    def test_full_exposure_pipeline(self, synth_env):
        """End-to-end: build grid → exposure → produces correct columns."""
        from backend.zones.build_zones import build_grid
        from backend.zones.exposure_join import compute_exposure

        zones = build_grid()
        zones = compute_exposure(zones, synth_env["flood_path"])

        required_cols = [
            "zone_id", "name", "flood_km2", "flood_pct",
            "population", "people_affected",
            "facilities_hit", "facilities_by_type",
            "road_cut_km", "confidence", "confidence_score", "reason",
        ]
        for col in required_cols:
            assert col in zones.columns, f"Missing column: {col}"

        # All remaining zones should have flood_pct >= MIN_FLOOD_PCT
        assert (zones["flood_pct"] >= config.MIN_FLOOD_PCT).all()


# ============================================================================
# Tests: Priority scoring
# ============================================================================

class TestPriorityScoring:

    def _make_feature(
        self,
        zone_id: str,
        flood_pct: float,
        people_affected: float,
        facilities_hit: int,
        road_cut_km: float,
        confidence: str = "Medium",
    ) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            "properties": {
                "zone_id": zone_id,
                "name": zone_id,
                "flood_pct": flood_pct,
                "people_affected": people_affected,
                "facilities_hit": facilities_hit,
                "road_cut_km": road_cut_km,
                "confidence": confidence,
                "confidence_score": 0.5,
                "reason": "",
            },
        }

    def _make_fc(self, features: list[dict]) -> dict:
        return {"type": "FeatureCollection", "features": features}

    def test_basic_scoring(self):
        from backend.priority.score import rescore

        fc = self._make_fc([
            self._make_feature("Z1", 80, 500, 3, 2.5),
            self._make_feature("Z2", 20, 100, 0, 0.5),
        ])
        result = rescore(fc)
        feats = result["features"]

        # Z1 should rank #1 (higher everything)
        assert feats[0]["properties"]["zone_id"] == "Z1"
        assert feats[0]["properties"]["rank"] == 1
        assert feats[1]["properties"]["rank"] == 2

        # Scores should be in (0, 1]
        for f in feats:
            assert 0 < f["properties"]["score"] <= 1.0

    def test_breakdown_present(self):
        from backend.priority.score import rescore

        fc = self._make_fc([
            self._make_feature("Z1", 50, 200, 1, 1.0),
        ])
        result = rescore(fc)
        bd = result["features"][0]["properties"]["breakdown"]
        assert set(bd.keys()) == {"severity", "people", "facilities", "roads"}

    def test_tier_assignment(self):
        """With 10 zones: 2 should be P1, 3 P2, 5 P3."""
        from backend.priority.score import rescore

        features = [
            self._make_feature(f"Z{i}", flood_pct=100 - i * 10, people_affected=1000 - i * 100,
                               facilities_hit=5 - i // 2, road_cut_km=5 - i * 0.5)
            for i in range(10)
        ]
        fc = self._make_fc(features)
        result = rescore(fc)

        tiers = [f["properties"]["tier"] for f in result["features"]]
        assert tiers.count("P1") == 2
        assert tiers.count("P2") == 3
        assert tiers.count("P3") == 5

    def test_verify_rule(self):
        """A P1/P2 zone with Low confidence → tier VERIFY."""
        from backend.priority.score import rescore

        fc = self._make_fc([
            self._make_feature("Z1", 90, 1000, 5, 5.0, confidence="Low"),
            self._make_feature("Z2", 10, 50, 0, 0.0, confidence="High"),
            self._make_feature("Z3", 5, 20, 0, 0.0, confidence="High"),
        ])
        result = rescore(fc)
        z1_props = result["features"][0]["properties"]

        assert z1_props["zone_id"] == "Z1"
        assert z1_props["tier"] == "VERIFY"
        assert "Low" in z1_props["reason"]

    def test_changing_weights_changes_ranking(self):
        """Different weights should produce different rankings."""
        from backend.priority.score import rescore

        fc = self._make_fc([
            self._make_feature("Z-roads", 10, 10, 0, 10.0),   # high roads only
            self._make_feature("Z-people", 10, 1000, 0, 0.0),  # high people only
        ])

        # Weight roads heavily
        r1 = rescore(fc, {"severity": 0.1, "people": 0.1, "facilities": 0.1, "roads": 0.7})
        assert r1["features"][0]["properties"]["zone_id"] == "Z-roads"

        # Weight people heavily
        r2 = rescore(fc, {"severity": 0.1, "people": 0.7, "facilities": 0.1, "roads": 0.1})
        assert r2["features"][0]["properties"]["zone_id"] == "Z-people"

    def test_rescore_empty_collection(self):
        from backend.priority.score import rescore
        fc = self._make_fc([])
        result = rescore(fc)
        assert result["features"] == []

    def test_rescore_with_zero_values(self):
        """Zones with all-zero values should score 0 and not crash."""
        from backend.priority.score import rescore

        fc = self._make_fc([
            self._make_feature("Z0", 0, 0, 0, 0.0),
        ])
        result = rescore(fc)
        assert result["features"][0]["properties"]["score"] == 0.0


# ============================================================================
# Tests: Integration (full pipeline write)
# ============================================================================

class TestIntegration:

    def test_zones_geojson_contract(self, synth_env):
        """Full pipeline → zones.geojson must have all contract fields."""
        from backend.zones.build_zones import run

        zones_path = run(force=True)
        assert zones_path.exists()

        with open(zones_path, "r", encoding="utf-8") as f:
            fc = json.load(f)

        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) > 0

        required_fields = [
            "zone_id", "name", "flood_pct", "flood_km2",
            "population", "people_affected",
            "facilities_hit", "road_cut_km",
            "score", "rank", "tier",
            "confidence", "confidence_score", "reason",
            "breakdown",
        ]
        for feat in fc["features"]:
            props = feat["properties"]
            for field in required_fields:
                assert field in props, f"Zone {props.get('zone_id')}: missing '{field}'"

    def test_flood_mask_geojson_written(self, synth_env):
        from backend.zones.build_zones import run
        run(force=True)
        fmask = synth_env["out_dir"] / "flood_mask.geojson"
        assert fmask.exists()

    def test_infrastructure_geojsons_written(self, synth_env):
        from backend.zones.build_zones import run
        run(force=True)

        fac = synth_env["out_dir"] / "facilities.geojson"
        roads = synth_env["out_dir"] / "roads_cut.geojson"
        assert fac.exists()
        assert roads.exists()
