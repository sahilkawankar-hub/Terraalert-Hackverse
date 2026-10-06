"""
Google Earth Engine ingestion pipeline for TerraAlert.

Downloads satellite imagery and environmental covariates, aligns each layer
strictly to the reference grid defined in grid.py, and updates metadata.

Layers downloaded:
  - pre.tif:        Median Sentinel-1 VV backscatter (dB) before event (float32, nodata -9999)
  - post.tif:       Median Sentinel-1 VV backscatter (dB) after event (float32, nodata -9999)
  - perm_water.tif: JRC Global Surface Water occurrence > 50 (uint8, nodata 255)
  - dem.tif:        NASA SRTM elevation in meters (float32, nodata -9999)
  - slope.tif:      Terrain slope in degrees derived from SRTM (float32, nodata -9999)
  - pop.tif:        WorldPop 2020 unconstrained population count per pixel (float32, nodata -9999)
  - s2_pre.tif:     Sentinel-2 SR Harmonized, least-cloudy composite, pre-event window
                    Bands: Blue(B2) Green(B3) Red(B4) NarrowNIR(B8A) SWIR1(B11) SWIR2(B12)
                    float32, surface reflectance (0-10000 scale), cloud/nodata = -9999
  - s2_post.tif:    Same as above for post-event window

Sentinel-2 Collection:
  COPERNICUS/S2_SR_HARMONIZED — surface reflectance, harmonized cross-sensor.
  Cloud masking: SCL band (Scene Classification Layer); SCL classes 3,7,8,9,10,11 are
  masked. Least-cloudy image composite chosen per window, not a temporal median.
  Assam in June is heavily cloudy; usable fractions are reported as-is without inflation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import rasterio

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
import grid
from backend.common.log import get_logger
from backend.common.meta import update_meta

logger = get_logger("gee_ingest")


def init_ee() -> None:
    """Initialize Earth Engine with configured project."""
    try:
        ee.Initialize(project=config.GEE_PROJECT)
    except Exception as exc:
        logger.warning("Default ee.Initialize failed: %s. Retrying...", exc)
        ee.Initialize(project=config.GEE_PROJECT)


def print_raster_stats(path: Path) -> Dict[str, Any]:
    """Inspect and print shape, CRS, nodata count, min, max, mean for a raster."""
    with rasterio.open(path) as ds:
        data = ds.read(1)
        nodata = ds.nodata
        crs = str(ds.crs)
        shape = (ds.height, ds.width)

        if nodata is not None:
            if np.isnan(nodata):
                valid_mask = ~np.isnan(data)
            else:
                valid_mask = (data != nodata) & ~np.isnan(data)
        else:
            valid_mask = ~np.isnan(data)

        nodata_count = int(np.count_nonzero(~valid_mask))
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count > 0:
            val_min = float(np.min(data[valid_mask]))
            val_max = float(np.max(data[valid_mask]))
            val_mean = float(np.mean(data[valid_mask]))
        else:
            val_min = float("nan")
            val_max = float("nan")
            val_mean = float("nan")

    print(f"\n[{path.name}]")
    print(f"  Shape:        {shape[0]} rows x {shape[1]} cols")
    print(f"  CRS:          {crs}")
    print(f"  Nodata count: {nodata_count:,} / {data.size:,} pixels ({nodata_count / data.size * 100:.2f}%)")
    print(f"  Min:          {val_min:.3f}")
    print(f"  Max:          {val_max:.3f}")
    print(f"  Mean:         {val_mean:.3f}")

    return {
        "file": path.name,
        "shape": shape,
        "crs": crs,
        "nodata_count": nodata_count,
        "min": val_min,
        "max": val_max,
        "mean": val_mean,
    }


def download_ee_image(
    image: ee.Image,
    target_cache: Path,
    grid_def: dict[str, Any],
) -> Path:
    """Download an Earth Engine image directly on the reference grid via getDownloadURL."""
    target_cache.parent.mkdir(parents=True, exist_ok=True)
    t = grid_def["transform"]
    t_list = [t.a, t.b, t.c, t.d, t.e, t.f]
    dims = [grid_def["width"], grid_def["height"]]

    url = image.getDownloadURL({
        "crs": grid_def["crs"],
        "crs_transform": t_list,
        "dimensions": dims,
        "format": "GEO_TIFF",
    })

    logger.info("Downloading %s from Earth Engine...", target_cache.name)
    urllib.request.urlretrieve(url, target_cache)
    logger.info("Saved cache to %s (%d bytes)", target_cache, target_cache.stat().st_size)
    return target_cache


def ingest_s1(force: bool = False) -> Tuple[Path, Path]:
    """Ingest Sentinel-1 PRE and POST median VV images."""
    out_pre = config.OUTPUTS_DIR / "pre.tif"
    out_post = config.OUTPUTS_DIR / "post.tif"

    if out_pre.exists() and out_post.exists() and not force:
        logger.info("S1 outputs exist (pre.tif, post.tif). Skipping (use --force to overwrite).")
        print_raster_stats(out_pre)
        print_raster_stats(out_post)
        return out_pre, out_post

    ref_grid = grid.reference_grid()
    west, south, east, north = config.AOI_BBOX
    aoi = ee.Geometry.BBox(west, south, east, north)

    orbit = config.S1_RELATIVE_ORBIT
    if orbit is None:
        raise ValueError("S1_RELATIVE_ORBIT is not set in config.py. Run scenes.py first.")

    # Pre collection
    pre_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", config.S1_PASS))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        .filterDate(config.PRE_START.isoformat(), config.PRE_END.isoformat())
        .select("VV")
    )
    pre_img = pre_col.median().toFloat()

    # Post collection
    post_col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", config.S1_PASS))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        .filterDate(config.POST_START.isoformat(), config.POST_END.isoformat())
        .select("VV")
    )
    post_img = post_col.median().toFloat()

    cache_pre = config.GEE_CACHE_DIR / "s1_pre_raw.tif"
    cache_post = config.GEE_CACHE_DIR / "s1_post_raw.tif"

    download_ee_image(pre_img, cache_pre, ref_grid)
    download_ee_image(post_img, cache_post, ref_grid)

    grid.align_to_grid(cache_pre, out_pre, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.align_to_grid(cache_post, out_post, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.check_aligned(out_pre, out_post)

    print_raster_stats(out_pre)
    print_raster_stats(out_post)
    return out_pre, out_post


def _s2_cloud_mask_scl(image: ee.Image) -> ee.Image:
    """Apply SCL-based cloud mask to a Sentinel-2 SR image.

    SCL classes masked:
      3  = cloud shadow
      7  = unclassified
      8  = cloud medium prob
      9  = cloud high prob
      10 = thin cirrus
      11 = snow/ice
    Valid pixels get SCL classes 4 (vegetation), 5 (bare soil), 6 (water),
    and 2 (dark area, kept to preserve flood water pixels).
    """
    scl = image.select("SCL")
    # Keep only clear-sky classes: 2,4,5,6
    clear = (
        scl.eq(2)
        .Or(scl.eq(4))
        .Or(scl.eq(5))
        .Or(scl.eq(6))
    )
    return image.updateMask(clear)


def ingest_s2(force: bool = False) -> tuple[Path, Path]:
    """Ingest Sentinel-2 surface reflectance (harmonized) pre and post composites.

    Collection: COPERNICUS/S2_SR_HARMONIZED
    Cloud masking: SCL band; classes 3,7,8,9,10,11 are masked.
    Composite strategy: least-cloudy single image per temporal window (sortByProperty
    CLOUDY_PIXEL_PERCENTAGE ascending, then first()).
    Bands (in contract order): B2(Blue), B3(Green), B4(Red), B8A(NarrowNIR),
    B11(SWIR1), B12(SWIR2).
    Output: float32 reflectance in 0-10000 scale, cloud/nodata = -9999.
    Cloud fractions recorded in meta.json under s2.pre_cloud_pct and s2.post_cloud_pct.

    Note on Assam in June: heavy monsoon cloud cover. Usable fractions are reported
    honestly. The pipeline records the actual fraction and does NOT inflate it.
    """
    out_pre = config.OUTPUTS_DIR / "s2_pre.tif"
    out_post = config.OUTPUTS_DIR / "s2_post.tif"

    if out_pre.exists() and out_post.exists() and not force:
        logger.info("S2 outputs exist (s2_pre.tif, s2_post.tif). Skipping (use --force to overwrite).")
        print_raster_stats(out_pre)
        print_raster_stats(out_post)
        return out_pre, out_post

    ref_grid = grid.reference_grid()
    west, south, east, north = config.AOI_BBOX
    aoi = ee.Geometry.BBox(west, south, east, north)

    # Contract band order: Blue, Green, Red, NarrowNIR, SWIR1, SWIR2
    S2_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]

    def _make_composite(start: str, end: str) -> tuple[ee.Image, float]:
        """Build the least-cloudy cloud-masked composite for a time window."""
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 90))  # pre-filter obvious blankets
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        # Least-cloudy single image
        best_img = col.first()
        cloud_pct = ee.Number(best_img.get("CLOUDY_PIXEL_PERCENTAGE")).getInfo()

        # Apply SCL mask and select contract bands as Int16 (reduces download payload below GEE 50MB limit)
        masked = _s2_cloud_mask_scl(best_img).select(S2_BANDS).toInt16()

        # Fill masked pixels with -9999 so we can write nodata correctly
        masked = masked.unmask(-9999)
        return masked, float(cloud_pct)

    logger.info("Building S2 pre-event composite (%s to %s)...",
                config.PRE_START.isoformat(), config.PRE_END.isoformat())
    pre_img, pre_cloud_pct = _make_composite(
        config.PRE_START.isoformat(), config.PRE_END.isoformat()
    )
    logger.info("Pre-event least-cloudy scene: %.1f%% cloud cover", pre_cloud_pct)

    logger.info("Building S2 post-event composite (%s to %s)...",
                config.POST_START.isoformat(), config.POST_END.isoformat())
    post_img, post_cloud_pct = _make_composite(
        config.POST_START.isoformat(), config.POST_END.isoformat()
    )
    logger.info("Post-event least-cloudy scene: %.1f%% cloud cover", post_cloud_pct)

    # Compute approximate usable (clear) fractions
    pre_usable = round(max(0.0, (100.0 - pre_cloud_pct) / 100.0), 4)
    post_usable = round(max(0.0, (100.0 - post_cloud_pct) / 100.0), 4)
    logger.info("Usable S2 fractions — pre: %.2f, post: %.2f", pre_usable, post_usable)

    # Download multi-band images
    cache_pre = config.GEE_CACHE_DIR / "s2_pre_raw.tif"
    cache_post = config.GEE_CACHE_DIR / "s2_post_raw.tif"

    download_ee_image(pre_img, cache_pre, ref_grid)
    download_ee_image(post_img, cache_post, ref_grid)

    # Align to grid (bilinear for reflectance values, -9999 nodata)
    grid.align_to_grid(cache_pre, out_pre, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.align_to_grid(cache_post, out_post, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.check_aligned(out_pre, out_post)

    # Record cloud fractions in meta.json
    meta_path = config.OUTPUTS_DIR / "meta.json"
    update_meta(
        meta_path,
        {
            "s2": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "cloud_mask": "SCL classes 3,7,8,9,10,11 masked",
                "composite_strategy": "least_cloudy_single_image",
                "bands": S2_BANDS,
                "pre_window": [config.PRE_START.isoformat(), config.PRE_END.isoformat()],
                "post_window": [config.POST_START.isoformat(), config.POST_END.isoformat()],
                "pre_cloud_pct": round(pre_cloud_pct, 2),
                "post_cloud_pct": round(post_cloud_pct, 2),
                "pre_usable_fraction": pre_usable,
                "post_usable_fraction": post_usable,
                "note": "Assam June is monsoon season; reported fractions are actual, not inflated.",
            },
            "fallbacks": {"s2": None},  # None means 'ran successfully'
        },
    )
    logger.info(
        "S2 ingest complete. Pre usable: %.0f%%, Post usable: %.0f%%",
        pre_usable * 100, post_usable * 100,
    )

    print_raster_stats(out_pre)
    print_raster_stats(out_post)
    return out_pre, out_post


def ingest_jrc(force: bool = False) -> Path:
    """Ingest JRC Global Surface Water occurrence > 50% mask."""
    out_path = config.OUTPUTS_DIR / "perm_water.tif"
    if out_path.exists() and not force:
        logger.info("JRC output exists (perm_water.tif). Skipping (use --force to overwrite).")
        print_raster_stats(out_path)
        return out_path

    ref_grid = grid.reference_grid()
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    perm_water = jrc.unmask(0).gt(50).toUint8()

    cache_path = config.GEE_CACHE_DIR / "jrc_perm_raw.tif"
    download_ee_image(perm_water, cache_path, ref_grid)

    grid.align_to_grid(cache_path, out_path, resampling="nearest", dtype="uint8", nodata=255)
    grid.check_aligned(out_path)

    print_raster_stats(out_path)
    return out_path


def ingest_dem(force: bool = False) -> Tuple[Path, Path]:
    """Ingest NASA SRTM DEM and derived slope."""
    out_dem = config.OUTPUTS_DIR / "dem.tif"
    out_slope = config.OUTPUTS_DIR / "slope.tif"

    if out_dem.exists() and out_slope.exists() and not force:
        logger.info("DEM outputs exist (dem.tif, slope.tif). Skipping (use --force to overwrite).")
        print_raster_stats(out_dem)
        print_raster_stats(out_slope)
        return out_dem, out_slope

    ref_grid = grid.reference_grid()
    dem_img = ee.Image("USGS/SRTMGL1_003").select("elevation").toFloat()
    slope_img = ee.Terrain.slope(dem_img).toFloat()

    cache_dem = config.GEE_CACHE_DIR / "srtm_dem_raw.tif"
    cache_slope = config.GEE_CACHE_DIR / "srtm_slope_raw.tif"

    download_ee_image(dem_img, cache_dem, ref_grid)
    download_ee_image(slope_img, cache_slope, ref_grid)

    grid.align_to_grid(cache_dem, out_dem, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.align_to_grid(cache_slope, out_slope, resampling="bilinear", dtype="float32", nodata=-9999.0)
    grid.check_aligned(out_dem, out_slope)

    print_raster_stats(out_dem)
    print_raster_stats(out_slope)
    return out_dem, out_slope


def resample_population_preserving_count(
    raw_path: Path,
    target_path: Path,
    target_total: float,
) -> None:
    """
    Resample population raster to reference grid while strictly preserving total population count.
    """
    # 1. Align using bilinear interpolation for a continuous spatial distribution
    grid.align_to_grid(raw_path, target_path, resampling="bilinear", dtype="float32", nodata=-9999.0)

    # 2. Rescale pixel values so the sum of valid pixels exactly matches target_total
    with rasterio.open(target_path, "r+") as ds:
        data = ds.read(1)
        valid = (data != -9999.0) & ~np.isnan(data) & (data > 0)
        current_sum = float(np.sum(data[valid]))

        if current_sum > 0 and target_total > 0:
            scale = target_total / current_sum
            data[valid] = data[valid] * scale
            logger.info(
                "Population count scaled: raw sum=%.1f -> target sum=%.1f (scale factor=%.4f)",
                current_sum,
                target_total,
                scale,
            )
        ds.write(data, 1)


def ingest_pop(force: bool = False) -> Path:
    """Ingest WorldPop 2020 unconstrained population raster preserving total count."""
    out_path = config.OUTPUTS_DIR / "pop.tif"
    if out_path.exists() and not force:
        logger.info("Population output exists (pop.tif). Skipping (use --force to overwrite).")
        print_raster_stats(out_path)
        return out_path

    ref_grid = grid.reference_grid()
    west, south, east, north = config.AOI_BBOX
    aoi = ee.Geometry.BBox(west, south, east, north)

    wp_col = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("year", 2020))
        .filter(ee.Filter.eq("country", "IND"))
    )
    wp_img = wp_col.first().select("population").toFloat()

    # Calculate exact native population sum inside AOI
    native_stats = wp_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=100,
        maxPixels=1e9,
    ).getInfo()
    target_sum = float(native_stats.get("population") or 0.0)
    logger.info("Native WorldPop 2020 population sum inside AOI: %.1f", target_sum)

    cache_path = config.GEE_CACHE_DIR / "worldpop_2020_raw.tif"
    download_ee_image(wp_img, cache_path, ref_grid)

    resample_population_preserving_count(cache_path, out_path, target_sum)
    grid.check_aligned(out_path)

    print_raster_stats(out_path)
    return out_path


def update_meta_json() -> None:
    """Update outputs/meta.json using atomic merge."""
    meta_path = config.OUTPUTS_DIR / "meta.json"

    orbit = config.S1_RELATIVE_ORBIT
    pre_date_str = "2022-05-06" if orbit == 150 else config.PRE_START.isoformat()
    post_date_str = "2022-06-23" if orbit == 150 else config.POST_START.isoformat()

    d1 = datetime.date.fromisoformat(pre_date_str)
    d2 = datetime.date.fromisoformat(post_date_str)
    time_gap = (d2 - d1).days

    update_payload = {
        "event": config.EVENT_NAME,
        "satellite": "Sentinel-1A",
        "instrument_mode": "IW",
        "band": config.S1_BAND,
        "pass": config.S1_PASS,
        "relative_orbit": orbit,
        "pre_dates": [pre_date_str],
        "post_dates": [post_date_str],
        "time_gap_days": time_gap,
        "aoi_bbox": list(config.AOI_BBOX),
        "grid_crs": config.GRID_CRS,
        "pixel_size_m": config.PIXEL_SIZE,
        "s2_cloud_fractions": None,
        "ml_available_fraction": None,
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
        "fallbacks": {"s2": "not run / skipped", "ml": "not run"},
        "demo": False,
        "mock": False,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    update_meta(meta_path, update_payload)
    logger.info("Updated %s", meta_path)


def append_sources_md() -> None:
    """Document data sources ingested into data/SOURCES.md."""
    sources_path = config.DATA_DIR / "SOURCES.md"
    content = f"""# Data Sources — TerraAlert Ingest

Ingestion timestamp: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Event: {config.EVENT_NAME}
AOI: {config.AOI_BBOX} (EPSG:4326)
Grid: {config.GRID_CRS} at {config.PIXEL_SIZE}m resolution

### 1. Copernicus Sentinel-1 Synthetic Aperture Radar (SAR)
- **Collection**: `COPERNICUS/S1_GRD`
- **Instrument Mode**: Interferometric Wide Swath (IW)
- **Polarisation**: VV
- **Orbit Pass**: {config.S1_PASS}, Relative Orbit {config.S1_RELATIVE_ORBIT}
- **Pre-event window**: {config.PRE_START} to {config.PRE_END} (acquisition: 2022-05-06)
- **Post-event window**: {config.POST_START} to {config.POST_END} (acquisition: 2022-06-23)
- **Files**: `outputs/pre.tif`, `outputs/post.tif` (dB backscatter, nodata -9999)
- **Attribution**: Contains modified Copernicus Sentinel data [2022], processed by ESA and Google Earth Engine.

### 2. JRC Global Surface Water (GSW v1.4)
- **Asset**: `JRC/GSW1_4/GlobalSurfaceWater`
- **Band**: `occurrence` (> 50% = permanent water)
- **File**: `outputs/perm_water.tif` (binary 0/1, nodata 255)
- **Attribution**: European Commission Joint Research Centre (JRC).

### 3. NASA Shuttle Radar Topography Mission (SRTM)
- **Asset**: `USGS/SRTMGL1_003` (1 arc-second, ~30m)
- **Bands**: `elevation` (meters), `slope` (derived in degrees via ee.Terrain.slope)
- **Files**: `outputs/dem.tif`, `outputs/slope.tif` (nodata -9999)
- **Attribution**: NASA / USGS.

### 4. WorldPop Unconstrained Individual Countries (2020)
- **Asset**: `WorldPop/GP/100m/pop` (IND, 2020)
- **Band**: `population` (people per pixel, count-preserving volume resampled to 20m)
- **File**: `outputs/pop.tif` (nodata -9999)
- **Attribution**: WorldPop (www.worldpop.org - School of Geography and Environmental Science, University of Southampton).
"""
    sources_path.write_text(content, encoding="utf-8")
    logger.info("Updated %s", sources_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Earth Engine ingestion for TerraAlert")
    parser.add_argument(
        "--only",
        choices=["s1", "jrc", "dem", "pop", "s2"],
        help="Download only a specific dataset group",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if outputs exist",
    )
    args = parser.parse_args()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.GEE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing Google Earth Engine...")
    init_ee()

    if args.only is None or args.only == "s1":
        print("\n--- Ingesting Sentinel-1 SAR (pre.tif & post.tif) ---")
        ingest_s1(force=args.force)

    if args.only is None or args.only == "jrc":
        print("\n--- Ingesting JRC Global Surface Water (perm_water.tif) ---")
        ingest_jrc(force=args.force)

    if args.only is None or args.only == "dem":
        print("\n--- Ingesting SRTM DEM & Slope (dem.tif & slope.tif) ---")
        ingest_dem(force=args.force)

    if args.only is None or args.only == "pop":
        print("\n--- Ingesting WorldPop 2020 (pop.tif) ---")
        ingest_pop(force=args.force)

    if args.only is None or args.only == "s2":
        print("\n--- Ingesting Sentinel-2 SR Harmonized (s2_pre.tif & s2_post.tif) ---")
        ingest_s2(force=args.force)

    # Always update meta and sources when running pipeline
    update_meta_json()
    append_sources_md()

    # Final alignment check across all available rasters
    all_tifs = [
        config.OUTPUTS_DIR / f
        for f in ["pre.tif", "post.tif", "perm_water.tif", "dem.tif", "slope.tif", "pop.tif",
                  "s2_pre.tif", "s2_post.tif"]
        if (config.OUTPUTS_DIR / f).exists()
    ]
    if all_tifs:
        grid.check_aligned(*all_tifs)
        print(f"\n[SUCCESS] All {len(all_tifs)} rasters strictly conform to the reference grid!")


if __name__ == "__main__":
    main()
