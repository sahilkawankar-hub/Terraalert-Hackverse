"""
Generate a realistic mock demo bundle in outputs/demo/ for the configured AOI.

Generates:
  - outputs/demo/zones.geojson       (~20 grid zones with all contract fields)
  - outputs/demo/flood_mask.geojson   (flood extent polygons)
  - outputs/demo/roads_cut.geojson    (cut road segments)
  - outputs/demo/facilities.geojson   (critical facilities with in_flood flag)
  - outputs/demo/meta.json            (event and pipeline metadata)
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.common.meta import update_meta
from backend.priority.score import rescore


def generate_mock_bundle(output_dir: Path | None = None) -> dict[str, Path]:
    """Generate all demo GeoJSON and metadata files in outputs/demo/."""
    random.seed(42)  # reproducible mock data
    if output_dir is None:
        output_dir = config.OUTPUTS_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = config.AOI_BBOX
    cols, rows = 5, 4  # 20 grid-like zones
    dx = (east - west) / cols
    dy = (north - south) / rows

    zone_features: list[dict[str, Any]] = []
    facility_features: list[dict[str, Any]] = []
    road_cut_features: list[dict[str, Any]] = []
    flood_polygon_features: list[dict[str, Any]] = []

    facility_types = ["hospital", "clinic", "school", "shelter", "water_plant"]
    road_types = ["primary", "secondary", "tertiary", "residential"]

    zone_idx = 1
    for r in range(rows):
        for c in range(cols):
            z_west = west + c * dx
            z_east = z_west + dx
            z_south = south + r * dy
            z_north = z_south + dy
            zone_id = f"ZONE-{zone_idx:02d}"

            poly_coords = [
                [
                    [round(z_west, 5), round(z_south, 5)],
                    [round(z_east, 5), round(z_south, 5)],
                    [round(z_east, 5), round(z_north, 5)],
                    [round(z_west, 5), round(z_north, 5)],
                    [round(z_west, 5), round(z_south, 5)],
                ]
            ]

            # Approximate area in km2 (at ~26 degrees latitude, 1 deg lat ~ 110.8 km, 1 deg lon ~ 100 km)
            approx_km2 = round((dx * 100.0) * (dy * 110.8), 2)

            # Plausible random values
            flood_pct = round(random.uniform(0.0, 45.0), 1)
            # Ensure at least some zones have significant flooding
            if zone_idx in (3, 7, 8, 12, 14, 18):
                flood_pct = round(random.uniform(35.0, 75.0), 1)
            elif zone_idx in (1, 5, 20):
                flood_pct = round(random.uniform(0.0, 3.0), 1)

            flood_km2 = round(approx_km2 * (flood_pct / 100.0), 2)
            population = int(random.randint(1200, 15000))
            people_affected = int(round(population * (flood_pct / 100.0) * random.uniform(0.6, 1.1)))
            people_affected = min(people_affected, population)

            # Facilities in zone
            num_fac = random.randint(1, 4)
            fac_by_type = {t: 0 for t in facility_types}
            fac_hit_count = 0

            for f_i in range(num_fac):
                f_type = random.choice(facility_types)
                f_lon = round(random.uniform(z_west + 0.1 * dx, z_east - 0.1 * dx), 5)
                f_lat = round(random.uniform(z_south + 0.1 * dy, z_north - 0.1 * dy), 5)
                # Likelihood of facility being flooded depends on zone flood percentage
                in_flood = (flood_pct > 15.0) and (random.random() < (flood_pct / 100.0 * 1.2))
                if in_flood:
                    fac_by_type[f_type] += 1
                    fac_hit_count += 1

                facility_features.append({
                    "type": "Feature",
                    "id": f"FAC-{zone_idx:02d}-{f_i+1}",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [f_lon, f_lat],
                    },
                    "properties": {
                        "facility_id": f"FAC-{zone_idx:02d}-{f_i+1}",
                        "name": f"{f_type.replace('_', ' ').title()} {zone_idx}-{f_i+1}",
                        "type": f_type,
                        "zone_id": zone_id,
                        "in_flood": in_flood,
                    },
                })

            # Roads cut in zone
            road_cut_km = 0.0
            if flood_pct > 10.0:
                num_cuts = random.randint(1, 3)
                for rc_i in range(num_cuts):
                    seg_len_km = round(random.uniform(0.3, 2.5), 2)
                    road_cut_km += seg_len_km
                    r_type = random.choice(road_types)
                    # Create a short LineString inside the zone
                    rx1 = round(random.uniform(z_west + 0.1 * dx, z_east - 0.2 * dx), 5)
                    ry1 = round(random.uniform(z_south + 0.1 * dy, z_north - 0.2 * dy), 5)
                    rx2 = round(rx1 + random.uniform(0.02 * dx, 0.15 * dx), 5)
                    ry2 = round(ry1 + random.uniform(0.02 * dy, 0.15 * dy), 5)

                    road_cut_features.append({
                        "type": "Feature",
                        "id": f"ROAD-{zone_idx:02d}-{rc_i+1}",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[rx1, ry1], [rx2, ry2]],
                        },
                        "properties": {
                            "road_id": f"ROAD-{zone_idx:02d}-{rc_i+1}",
                            "name": f"{r_type.title()} Road {zone_idx}",
                            "road_type": r_type,
                            "length_km": seg_len_km,
                            "zone_id": zone_id,
                            "status": "submerged",
                        },
                    })
            road_cut_km = round(road_cut_km, 2)

            # Plausible flood polygons for flooded zones
            if flood_pct > 5.0:
                fw = z_west + 0.1 * dx
                fe = fw + dx * (flood_pct / 100.0) * 0.9
                fs = z_south + 0.1 * dy
                fn = fs + dy * (flood_pct / 100.0) * 0.9
                flood_polygon_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [round(fw, 5), round(fs, 5)],
                                [round(fe, 5), round(fs, 5)],
                                [round(fe, 5), round(fn, 5)],
                                [round(fw, 5), round(fn, 5)],
                                [round(fw, 5), round(fs, 5)],
                            ]
                        ],
                    },
                    "properties": {
                        "zone_id": zone_id,
                        "area_km2": flood_km2,
                        "source": "SAR_Sentinel1",
                    },
                })

            # Confidence determination
            conf_score = round(random.uniform(0.35, 0.95), 2)
            # Give specific zones Low confidence to exercise the VERIFY rule
            if zone_idx in (7, 14):
                conf_score = 0.38
                confidence = "Low"
            elif conf_score >= config.CONF_HIGH:
                confidence = "High"
            elif conf_score <= config.CONF_LOW:
                confidence = "Low"
            else:
                confidence = "Medium"

            reason = (
                f"{flood_pct:.1f}% flood extent ({flood_km2} km²), {people_affected} affected, "
                f"{fac_hit_count} critical facilities inundated"
            )

            zone_features.append({
                "type": "Feature",
                "id": zone_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": poly_coords,
                },
                "properties": {
                    "zone_id": zone_id,
                    "name": f"Barpeta Sector {c+1}-{r+1}",
                    "flood_pct": flood_pct,
                    "flood_km2": flood_km2,
                    "population": population,
                    "people_affected": people_affected,
                    "facilities_hit": fac_hit_count,
                    "road_cut_km": road_cut_km,
                    "confidence": confidence,
                    "confidence_score": conf_score,
                    "reason": reason,
                    "facilities_by_type": fac_by_type,
                    "breakdown": {
                        "severity": 0.0,
                        "people": 0.0,
                        "facilities": 0.0,
                        "roads": 0.0,
                    },
                    "score": 0.0,
                    "rank": 0,
                    "tier": "P3",
                },
            })
            zone_idx += 1

    # Score and rank zones using the official scoring logic
    zone_fc = {"type": "FeatureCollection", "features": zone_features}
    ranked_fc = rescore(zone_fc, config.WEIGHTS)

    # File paths
    zones_path = output_dir / "zones.geojson"
    flood_path = output_dir / "flood_mask.geojson"
    roads_path = output_dir / "roads_cut.geojson"
    facilities_path = output_dir / "facilities.geojson"
    meta_path = output_dir / "meta.json"

    # Write GeoJSON files
    with open(zones_path, "w", encoding="utf-8") as f:
        json.dump(ranked_fc, f, indent=2)

    with open(flood_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": flood_polygon_features}, f, indent=2)

    with open(roads_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": road_cut_features}, f, indent=2)

    with open(facilities_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": facility_features}, f, indent=2)

    # Build meta.json adhering to contract
    meta_payload = {
        "event": config.EVENT_NAME,
        "satellite": "Sentinel-1 (C-SAR)",
        "pass": config.S1_PASS,
        "relative_orbit": config.S1_RELATIVE_ORBIT,
        "pre_dates": [config.PRE_START.isoformat(), config.PRE_END.isoformat()],
        "post_dates": [config.POST_START.isoformat(), config.POST_END.isoformat()],
        "time_gap_days": (config.POST_START - config.PRE_END).days,
        "s2_cloud_fraction": 0.12,
        "ml_available_fraction": 0.0,
        "model_name": config.ML_MODEL,
        "model_fallback": config.ML_MODEL_FALLBACK,
        "thresholds": {
            "post_db_max": config.POST_DB_MAX,
            "diff_db_max": config.DIFF_DB_MAX,
            "slope_max_deg": config.SLOPE_MAX_DEG,
            "min_object_pixels": config.MIN_OBJECT_PIXELS,
            "min_flood_pct": config.MIN_FLOOD_PCT,
        },
        "weights": config.WEIGHTS,
        "tier_fractions": {
            "p1": config.TIER_P1_FRAC,
            "p2": config.TIER_P2_FRAC,
        },
        "demo": True,
        "mock": True,
        "fallbacks": {
            "ml": "not run",
        },
        "aoi_bbox": list(config.AOI_BBOX),
        "grid_crs": config.GRID_CRS,
        "pixel_size_m": config.PIXEL_SIZE,
    }

    update_meta(meta_path, meta_payload)

    return {
        "zones": zones_path,
        "flood": flood_path,
        "roads": roads_path,
        "facilities": facilities_path,
        "meta": meta_path,
    }


if __name__ == "__main__":
    paths = generate_mock_bundle()
    print("Mock demo bundle generated successfully:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
