"""
Sentinel-1 scene listing and orbit selection for TerraAlert.

Queries Earth Engine COPERNICUS/S1_GRD over the AOI for PRE and POST event
windows, identifies common pass + relative orbit combinations, recommends
the optimal pair, and updates config.py.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ee

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger

logger = get_logger("scenes")


def init_ee() -> None:
    """Initialize Earth Engine with configured project."""
    try:
        ee.Initialize(project=config.GEE_PROJECT)
    except Exception as exc:
        logger.warning("Default ee.Initialize failed: %s. Trying with config project...", exc)
        try:
            ee.Initialize(project=config.GEE_PROJECT)
        except Exception as e:
            logger.error("Failed to initialize Earth Engine: %s", e)
            raise


def get_s1_scenes(
    bbox: tuple[float, float, float, float],
    start_date: datetime.date,
    end_date: datetime.date,
) -> List[Dict[str, Any]]:
    """
    Query S1_GRD collection for IW mode with VV polarization over bbox and date range.
    """
    west, south, east, north = bbox
    aoi = ee.Geometry.BBox(west, south, east, north)
    
    # End date is inclusive in user intent, filterDate end is exclusive in EE
    ee_end = end_date + datetime.timedelta(days=1)
    
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filterDate(start_date.isoformat(), ee_end.isoformat())
        .sort("system:time_start")
    )
    
    # Extract properties
    def extract_info(img):
        return ee.Feature(
            None,
            {
                "id": img.id(),
                "time_start": img.get("system:time_start"),
                "pass": img.get("orbitProperties_pass"),
                "relativeOrbit": img.get("relativeOrbitNumber_start"),
                "platform": img.get("platformSerializerId"),
            },
        )
    
    features = col.map(extract_info).getInfo().get("features", [])
    scenes = []
    for f in features:
        props = f.get("properties", {})
        ts_ms = props.get("time_start")
        dt_str = "Unknown"
        if ts_ms:
            dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc)
            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Derive platform from id if platformSerializerId is None
        scene_id = props.get("id", "")
        platform = props.get("platform") or ("S1A" if "S1A" in scene_id else ("S1B" if "S1B" in scene_id else "S1"))

        scenes.append({
            "id": scene_id,
            "datetime": dt_str,
            "date": dt_str.split(" ")[0] if " " in dt_str else dt_str,
            "platform": platform,
            "pass": props.get("pass"),
            "relative_orbit": props.get("relativeOrbit"),
        })
    return scenes


def calculate_aoi_coverage(
    bbox: tuple[float, float, float, float],
    pass_direction: str,
    relative_orbit: int,
) -> float:
    """Compute the percentage of AOI covered by images from this pass and orbit."""
    west, south, east, north = bbox
    aoi = ee.Geometry.BBox(west, south, east, north)
    
    constant = ee.Image(1).clip(aoi)
    total_pix = constant.reduceRegion(ee.Reducer.count(), aoi, scale=100).getInfo().get("constant", 1)
    
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", pass_direction))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filterDate("2022-04-20", "2022-07-01")
    )
    mosaic = col.select("VV").median()
    stats = mosaic.reduceRegion(ee.Reducer.count(), aoi, scale=100).getInfo()
    cnt = stats.get("VV", 0)
    return round((cnt / total_pix) * 100.0, 1) if total_pix else 0.0


def find_matching_combinations(
    bbox: tuple[float, float, float, float],
    pre_scenes: List[Dict[str, Any]],
    post_scenes: List[Dict[str, Any]],
) -> List[Tuple[str, int, int, int, float]]:
    """
    Find (pass, relative_orbit) combinations in both PRE and POST windows.
    Returns list of (pass, relative_orbit, pre_count, post_count, coverage_pct).
    """
    pre_combos: Dict[Tuple[str, int], int] = {}
    for s in pre_scenes:
        p, o = s["pass"], s["relative_orbit"]
        if p and o is not None:
            pre_combos[(p, o)] = pre_combos.get((p, o), 0) + 1

    post_combos: Dict[Tuple[str, int], int] = {}
    for s in post_scenes:
        p, o = s["pass"], s["relative_orbit"]
        if p and o is not None:
            post_combos[(p, o)] = post_combos.get((p, o), 0) + 1

    common = []
    for key in set(pre_combos.keys()) & set(post_combos.keys()):
        cov = calculate_aoi_coverage(bbox, key[0], key[1])
        common.append((key[0], key[1], pre_combos[key], post_combos[key], cov))
    
    # Sort by coverage descending, then total scene count descending
    common.sort(key=lambda x: (x[4], x[2] + x[3]), reverse=True)
    return common


def print_scene_table(title: str, scenes: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 75}")
    print(f" {title} ({len(scenes)} scenes)")
    print(f"{'=' * 75}")
    header = f"{'Date':<12} {'Time (UTC)':<10} {'Platform':<10} {'Pass':<12} {'Rel Orbit':<10} {'Scene ID'}"
    print(header)
    print("-" * 75)
    for s in scenes:
        dt_parts = s["datetime"].split(" ")
        d_str = dt_parts[0] if len(dt_parts) > 0 else s["date"]
        t_str = dt_parts[1] if len(dt_parts) > 1 else ""
        print(
            f"{d_str:<12} {t_str:<10} {s['platform']:<10} {str(s['pass']):<12} "
            f"{str(s['relative_orbit']):<10} {s['id']}"
        )


def update_config_orbit(pass_direction: str, relative_orbit: int) -> None:
    """Update S1_PASS and S1_RELATIVE_ORBIT in config.py."""
    config_path = ROOT_DIR / "config.py"
    content = config_path.read_text(encoding="utf-8")
    
    # Replace S1_PASS
    content = re.sub(
        r'S1_PASS:\s*str\s*=\s*"[^"]*"',
        f'S1_PASS: str = "{pass_direction}"',
        content,
    )
    # Replace S1_RELATIVE_ORBIT
    content = re.sub(
        r'S1_RELATIVE_ORBIT:\s*Optional\[int\]\s*=\s*.*',
        f'S1_RELATIVE_ORBIT: Optional[int] = {relative_orbit}',
        content,
    )
    config_path.write_text(content, encoding="utf-8")
    logger.info("Updated config.py: S1_PASS=%s, S1_RELATIVE_ORBIT=%d", pass_direction, relative_orbit)


def aoi_sanity_check(
    pass_direction: str,
    relative_orbit: int,
) -> Dict[str, Any]:
    """
    Run coarse-scale AOI sanity check:
    - Median VV for pre and post windows
    - Flood share: share where post < -18 dB and (post - pre) < -3 dB (excluding JRC > 50)
    - Pre-existing water share and permanent water share
    """
    west, south, east, north = config.AOI_BBOX
    aoi = ee.Geometry.BBox(west, south, east, north)

    pre_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", pass_direction))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filterDate(config.PRE_START.isoformat(), config.PRE_END.isoformat())
        .select("VV")
    )

    post_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", pass_direction))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filterDate(config.POST_START.isoformat(), config.POST_END.isoformat())
        .select("VV")
    )

    pre_vv = pre_col.median()
    post_vv = post_col.median()
    diff_vv = post_vv.subtract(pre_vv)

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    not_perm_water = jrc.lte(50)

    # Flood signal: post < -18 dB and (post - pre) < -3 dB, excluding JRC occurrence > 50
    flood = post_vv.lt(config.POST_DB_MAX).And(diff_vv.lt(config.DIFF_DB_MAX)).And(not_perm_water)
    pre_water = pre_vv.lt(config.POST_DB_MAX).And(not_perm_water)
    perm_water = jrc.gt(50)

    stats = ee.Dictionary({
        "flood_share": flood.reduceRegion(ee.Reducer.mean(), aoi, scale=100, maxPixels=1e9).get("VV"),
        "pre_water_share": pre_water.reduceRegion(ee.Reducer.mean(), aoi, scale=100, maxPixels=1e9).get("VV"),
        "perm_water_share": perm_water.reduceRegion(ee.Reducer.mean(), aoi, scale=100, maxPixels=1e9).get("occurrence"),
        "pre_vv_median": pre_vv.reduceRegion(ee.Reducer.median(), aoi, scale=100, maxPixels=1e9).get("VV"),
        "post_vv_median": post_vv.reduceRegion(ee.Reducer.median(), aoi, scale=100, maxPixels=1e9).get("VV"),
    }).getInfo()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel-1 scene listing and orbit selection")
    parser.add_argument("--update-config", action="store_true", default=True, help="Update config.py with recommended orbit")
    parser.add_argument("--check-sanity", action="store_true", default=True, help="Run AOI sanity check")
    args = parser.parse_args()

    print(f"Initializing Earth Engine (project: {config.GEE_PROJECT})...")
    init_ee()

    print(f"AOI: {config.AOI_BBOX}")
    print(f"PRE Window:  {config.PRE_START} to {config.PRE_END}")
    print(f"POST Window: {config.POST_START} to {config.POST_END}")

    pre_scenes = get_s1_scenes(config.AOI_BBOX, config.PRE_START, config.PRE_END)
    post_scenes = get_s1_scenes(config.AOI_BBOX, config.POST_START, config.POST_END)

    print_scene_table("PRE-EVENT SCENES", pre_scenes)
    print_scene_table("POST-EVENT SCENES", post_scenes)

    matches = find_matching_combinations(config.AOI_BBOX, pre_scenes, post_scenes)

    print(f"\n{'-' * 80}")
    print(" Pass + Orbit Combinations appearing in BOTH windows:")
    print(f"{'-' * 80}")
    if not matches:
        print(" [!] No common pass + relative orbit found in both windows.")
        print(" Suggestion: Widen windows (e.g. PRE_START=2022-04-01, POST_END=2022-07-05).")
        return

    for p, o, n_pre, n_post, cov in matches:
        print(f"  * Pass: {p:<12} | Relative Orbit: {o:<4} | Pre scenes: {n_pre} | Post scenes: {n_post} | AOI Coverage: {cov:>5.1f}%")

    # Prefer 100% AOI coverage and DESCENDING (as pre-configured)
    full_cov = [m for m in matches if m[4] >= 99.0]
    desc_full = [m for m in full_cov if m[0] == "DESCENDING"]
    if desc_full:
        recommended = desc_full[0]
    elif full_cov:
        recommended = full_cov[0]
    else:
        recommended = matches[0]

    rec_pass, rec_orbit = recommended[0], recommended[1]
    
    print(f"\n>>> RECOMMENDATION: S1_PASS='{rec_pass}', S1_RELATIVE_ORBIT={rec_orbit}")
    print(f"    Selected for {recommended[4]}% AOI coverage and consistent {rec_pass} geometry")
    print(f"    with {recommended[2]} pre scene(s) and {recommended[3]} post scene(s).\n")

    if args.update_config:
        update_config_orbit(rec_pass, rec_orbit)
        print(f"Updated config.py -> S1_PASS = \"{rec_pass}\", S1_RELATIVE_ORBIT = {rec_orbit}")

    if args.check_sanity:
        print("\n" + "=" * 80)
        print(" AOI SANITY CHECK (Step 2)")
        print("=" * 80)
        stats = aoi_sanity_check(rec_pass, rec_orbit)
        flood_pct = (stats.get("flood_share") or 0.0) * 100.0
        perm_pct = (stats.get("perm_water_share") or 0.0) * 100.0
        pre_water_pct = (stats.get("pre_water_share") or 0.0) * 100.0
        pre_med = stats.get("pre_vv_median")
        post_med = stats.get("post_vv_median")

        print(f" Orbit: {rec_pass} Orbit {rec_orbit}")
        print(f" Pre-event median VV:         {pre_med:.2f} dB" if pre_med is not None else " Pre-event median VV: N/A")
        print(f" Post-event median VV:        {post_med:.2f} dB" if post_med is not None else " Post-event median VV: N/A")
        print(f" Permanent water (JRC > 50%): {perm_pct:.2f}% of AOI")
        print(f" Pre-window water share:      {pre_water_pct:.2f}% of AOI")
        print(f" New Flood signal share:      {flood_pct:.2f}% of AOI")
        print("-" * 80)

        if flood_pct >= 2.0:
            print(f" [PASS] Strong flood signal detected: {flood_pct:.2f}% >= 2.0% threshold.")
        else:
            print(f" [WARN] Weak flood signal: {flood_pct:.2f}% < 2.0% threshold.")
            print(" Proposed alternatives:")
            print("  1. Barpeta central/west: (90.80, 26.15, 91.10, 26.42)")
            print("  2. Kamrup / upstream:   (91.20, 26.05, 91.50, 26.32)")


if __name__ == "__main__":
    main()
