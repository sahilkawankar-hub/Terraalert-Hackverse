# TerraAlert context (read before any task)

Goal: compare PRE-event and POST-event satellite data, detect change caused by a disaster,
and produce a ranked PRIORITY MAP for emergency responders, with a confidence score and a
reason per zone, and an incident report. Prototype event: Assam floods, June 2022.
No login or authentication anywhere.

SOLO MODE: one engineer works on backend and AI; a separate person builds the frontend.
Keep work lean. Use the simplest approach that satisfies the contract. Use grid zones, skip
anything marked optional, and do not add features that were not asked for. If a step will take
more than about an hour of agent work, say so first and propose a simpler alternative.

HARD RULES
1. True pre/post comparison is mandatory. A single-image classifier is NOT acceptable.
   Any single-image model must run on BOTH pre and post and be differenced:
   flood = water now AND NOT water before AND NOT permanent water.
2. Every zone has a confidence label and a human-readable reason.
3. Every number in the UI and report is computed from data. Never LLM-generated.
4. Never fabricate data, metrics or model outputs. If something is unavailable, say so, use the
   documented fallback, and record it in outputs/meta.json under "fallbacks".
5. Sen1Floods11 labels: 1 water, 0 not water, -1 no data. ALWAYS mask -1 in metrics.
   Sen1Floods11 has ONLY post-event images and covers specific chips, not necessarily our AOI.
   Never claim accuracy on the AOI unless a reference actually overlaps it.
6. Measure area and length only in a metric CRS, never in degrees.

SETTINGS (live in config.py only; no hardcoded constants elsewhere)
- EVENT_NAME = "Assam Floods, June 2022 (Barpeta-Nalbari, Brahmaputra north bank)"
- AOI_BBOX = (90.95, 26.10, 91.25, 26.37)   # west, south, east, north, EPSG:4326
- PRE_START/PRE_END = 2022-04-20 / 2022-05-10 ; POST_START/POST_END = 2022-06-17 / 2022-06-30
  (windows are wide because one Sentinel-1 satellite repeats an orbit every 12 days; the pre
  window may overlap a minor earlier flood, so check it)
- GEE_PROJECT = "<<YOUR_CLOUD_PROJECT_ID>>"
- GRID_CRS = "EPSG:32646", PIXEL_SIZE = 20
- S1_PASS = "DESCENDING", S1_RELATIVE_ORBIT = None (set after listing scenes), S1_BAND = "VV"
- POST_DB_MAX=-18.0, DIFF_DB_MAX=-3.0, SLOPE_MAX_DEG=5.0, STEEP_FLAG_DEG=15.0,
  MIN_OBJECT_PIXELS=25, MIN_FLOOD_PCT=1.0
- WEIGHTS = severity 0.30, people 0.30, facilities 0.20, roads 0.20
- TIER_P1_FRAC=0.20, TIER_P2_FRAC=0.30
- CONF_WEIGHTS = agreement 0.5, terrain 0.3, time 0.2 ; CONF_HIGH=0.7, CONF_LOW=0.4
- ZONE_SOURCE = "grid", ZONE_CELL_M = 1000
- DEMO_MODE via env var DEMO_MODE=1
- ML_MODEL = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
  ML_MODEL_FALLBACK = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11"
  ML_DEVICE = "auto", ML_TILE = 512, ML_OVERLAP = 64, ML_PROB_HIGH = 0.8
  FUSION_RULE = "classical_plus_ml"

REFERENCE GRID: all rasters share one grid (GRID_CRS, PIXEL_SIZE, origin snapped to a multiple
of the pixel size, built from AOI_BBOX). grid.py provides reference_grid(),
align_to_grid(src, dst, resampling, dtype, nodata) (bilinear for continuous data, nearest for
masks) and check_aligned(*paths), which raises if a file is off-grid. Call check_aligned at the
start of every stage.

FILE CONTRACT (outputs/)
pre.tif, post.tif         float32 S1 VV in dB, nodata -9999
perm_water.tif            uint8, 1 = JRC occurrence>50, nodata 255
dem.tif, slope.tif        float32, nodata -9999
s2_pre.tif, s2_post.tif   float32 reflectance, bands: Blue B2, Green B3, Red B4,
                          NarrowNIR B8A, SWIR1 B11, SWIR2 B12; cloud = -9999
pop.tif                   float32 people per pixel (count preserved when resampling)
flood_classical.tif       uint8, 1 = new flood, nodata 255
flood_ml.tif              uint8, 1 = new flood, 255 = optical unusable
flood_fused.tif           uint8 final flood mask
agreement.tif             uint8: 0 none, 1 one method, 2 both, 255 nodata
confidence_pixel.tif      float32 0-1
meta.json                 event, satellite, pass, relative_orbit, pre_dates, post_dates,
                          time_gap_days, s2 cloud fractions, ml_available_fraction, model
                          names/versions, thresholds, metrics, fallbacks, demo flag
zones.geojson (EPSG:4326) zone_id, name, flood_pct, flood_km2, population, people_affected,
                          facilities_hit, road_cut_km, score, rank, tier (P1|P2|P3|VERIFY),
                          confidence (High|Medium|Low), confidence_score, reason,
                          breakdown (dict of factor contributions), facilities_by_type (dict)
flood_mask.geojson, roads_cut.geojson, facilities.geojson (EPSG:4326, facilities have in_flood)
overlays/{pre,post,flood,confidence}.png + overlays/bounds.json
trap_comparison.png, report.html, report.pdf
Do not change file names or field names without telling the user.

ENGINEERING RULES
- Python modules with CLIs, type hints, docstrings, logging (not print), pytest tests on
  synthetic data with no network. Skip work whose output exists unless --force.
- Verify every external dataset ID, API and package version before using it; catalogs change.
- Work only in the folders the task names. Stop after each phase and wait for "continue".
- Large rasters stay out of git (data/raw/, outputs/*.tif); outputs/demo/ is committed.
