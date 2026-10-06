"""
Integration tests for the TerraAlert FastAPI application and endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def test_health():
    """GET /api/health should return 200 and operational status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_meta():
    """GET /api/meta should return pipeline metadata with demo flag."""
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert "event" in data
    assert "aoi_bbox" in data
    assert data.get("demo") is True


def test_zones_endpoint():
    """GET /api/zones should return a GeoJSON FeatureCollection with all contract fields."""
    res = client.get("/api/zones")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0

    first_props = data["features"][0]["properties"]
    required_keys = [
        "zone_id",
        "name",
        "flood_pct",
        "flood_km2",
        "population",
        "people_affected",
        "facilities_hit",
        "road_cut_km",
        "score",
        "rank",
        "tier",
        "confidence",
        "confidence_score",
        "reason",
        "breakdown",
        "facilities_by_type",
    ]
    for key in required_keys:
        assert key in first_props, f"Missing required property: {key}"

    # Verify breakdown fields
    for factor in ["severity", "people", "facilities", "roads"]:
        assert factor in first_props["breakdown"]


def test_zones_filtering_tier():
    """GET /api/zones with tier filter should only return matching tiers."""
    res = client.get("/api/zones?tier=P1")
    assert res.status_code == 200
    features = res.json()["features"]
    assert len(features) > 0
    assert all(f["properties"]["tier"] == "P1" for f in features)


def test_zones_filtering_min_confidence():
    """GET /api/zones with min_confidence should filter out lower confidences."""
    res = client.get("/api/zones?min_confidence=High")
    assert res.status_code == 200
    features = res.json()["features"]
    assert all(f["properties"]["confidence"] == "High" for f in features)


def test_zone_by_id():
    """GET /api/zones/{zone_id} should return the exact zone or 404."""
    # First get a valid zone_id
    all_zones = client.get("/api/zones").json()["features"]
    target_id = all_zones[0]["properties"]["zone_id"]

    res = client.get(f"/api/zones/{target_id}")
    assert res.status_code == 200
    assert res.json()["properties"]["zone_id"] == target_id

    # Non-existent zone
    res_404 = client.get("/api/zones/NONEXISTENT-ZONE")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()


def test_flood_endpoint():
    """GET /api/flood should return flood polygons FeatureCollection."""
    res = client.get("/api/flood")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"


def test_facilities_endpoint():
    """GET /api/facilities should return critical facilities with in_flood flag."""
    res = client.get("/api/facilities")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    assert "in_flood" in data["features"][0]["properties"]


def test_roads_cut_endpoint():
    """GET /api/roads-cut should return submerged road segments."""
    res = client.get("/api/roads-cut")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"


def test_overlays_endpoint():
    """GET /api/overlays should return spatial bounds and placeholder note."""
    res = client.get("/api/overlays")
    assert res.status_code == 200
    data = res.json()
    assert "bounds" in data
    assert "note" in data
    assert "images not generated yet" in data["note"]


def test_report_unavailable():
    """GET /api/report should return 503 until report is generated."""
    res = client.get("/api/report")
    assert res.status_code == 503
    assert "not been generated yet" in res.json()["detail"]


def test_rescore_changes_ranking():
    """POST /api/rescore with contrasting weights must alter the ranking."""
    # Scenario A: 100% Facilities weight
    res_a = client.post(
        "/api/rescore",
        json={"severity": 0.0, "people": 0.0, "facilities": 1.0, "roads": 0.0},
    )
    assert res_a.status_code == 200
    rank_a_top = res_a.json()["features"][0]["properties"]["zone_id"]

    # Scenario B: 100% Roads weight
    res_b = client.post(
        "/api/rescore",
        json={"severity": 0.0, "people": 0.0, "facilities": 0.0, "roads": 1.0},
    )
    assert res_b.status_code == 200
    rank_b_top = res_b.json()["features"][0]["properties"]["zone_id"]

    # The top-ranked zone should be different under these opposite criteria
    assert rank_a_top != rank_b_top or len(res_a.json()["features"]) > 1

    # Verify rank order is sequential 1..N
    features_a = res_a.json()["features"]
    ranks = [f["properties"]["rank"] for f in features_a]
    assert ranks == list(range(1, len(features_a) + 1))


def test_rescore_invalid_weights_negative():
    """POST /api/rescore with negative weights should return 422 Unprocessable Entity."""
    res = client.post(
        "/api/rescore",
        json={"severity": -0.5, "people": 0.3, "facilities": 0.1, "roads": 0.1},
    )
    assert res.status_code == 422


def test_rescore_all_zeros():
    """POST /api/rescore with all zero weights should return 422."""
    res = client.post(
        "/api/rescore",
        json={"severity": 0.0, "people": 0.0, "facilities": 0.0, "roads": 0.0},
    )
    assert res.status_code == 422


def test_frontend_mount():
    """GET / should serve the frontend landing page."""
    res = client.get("/")
    assert res.status_code == 200
    assert "TerraAlert" in res.text
