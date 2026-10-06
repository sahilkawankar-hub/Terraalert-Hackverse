# TerraAlert Integration Test Report

**Execution Timestamp:** 2026-10-06 09:55 UTC  
**Target Branch:** `main` (commit `af8391d`)  
**Environment:** Windows, Python 3.12.10, virtual environment active  

---

## 1. Executive Summary & Verification Matrix

| Check / Subsystem | Status | Evidence / Observations |
|---|---|---|
| **Git & Working Tree** | **PASS** | `main` branch clean at commit `af8391d`. |
| **Requirements & Dependencies** | **PASS** (with caveats) | Core `requirements.txt` installed and functional. `weasyprint` not installed (requires system Pango/Cairo DLLs on Windows). |
| **Full Pipeline (`run_pipeline --force`)** | **FAIL** | Ingest (88.3s), Detect (1.7s), Fuse (0.97s), Zones (9.3s) succeeded. Priority failed (0.36s) due to unrecognized `--force` argument in `backend/priority/score.py`. ML stage skipped. |
| **ML Module Integration** | **FAIL** | `backend/ml/` contains `prithvi.py` and `temporal.py` but no pipeline entry point (`infer.py`/`__main__.py`). `flood_ml.tif`, `water_pre.tif`, and `water_post.tif` are 100% all zero values. |
| **Test Suite (`pytest`)** | **FAIL** | 83 passed, 7 skipped, 5 failed. All 5 failures in `tests/test_api.py` (`population` float vs int schema mismatch and `meta.demo` assertion). |
| **Raster Grid Alignment (`grid.check_aligned`)** | **PASS** | All 17 rasters in `outputs/` share identical grid parameters (1521 × 1475, EPSG:32646, 20m res, identical bounds). |
| **Sentinel-1 dB Radiometry** | **PASS** | `pre.tif`: [-33.9 dB, +26.2 dB], 652 px (0.03%) outside -40..+10 dB. `post.tif`: [-32.7 dB, +26.3 dB], 1577 px (0.07%) outside range. |
| **Permanent Water Separation** | **FAIL** | 3,698 pixels flagged in `flood_classical.tif` and `flood_fused.tif`; 100% of detected flood pixels overlap with `perm_water.tif` (Brahmaputra river channel). |
| **Zone GeoJSON Schema & Attributes** | **PASS** | 41 grid zones created; all 16 required attributes present; tier distribution: P1=5, P2=6, P3=19, VERIFY=11; confidence: Medium=19, Low=22. |
| **FastAPI Core Endpoints** | **FAIL** | `/api/zones`, `/api/zones/{id}`, and `POST /api/rescore` return **HTTP 500** due to Pydantic validation (`population: int` vs float values). `/api/health`, `/api/meta`, `/api/flood`, `/api/facilities`, `/api/roads-cut`, `/api/overlays/*` return **HTTP 200**. |
| **Overlays Service** | **PASS** | PNG overlays generated at 1615×1414 with `bounds.json` matching AOI extent [90.95, 26.1, 91.25, 26.37]. |
| **HTML / PDF Reporting** | **PASS** | `report.html` generated with summary KPIs. PDF generation gracefully skipped with notice when WeasyPrint is absent. |
| **Frontend Map & Overlays** | **PASS** | Leaflet-based frontend loads GeoJSON and displays fallback data gracefully. (Frontend uses Leaflet, not MapLibre; renders vector GeoJSON, not raster tiles). |

---

## 2. Issues & Defect Classification

### MUST FIX (Blocking bugs)
1. **Pydantic Schema Validation Error (HTTP 500 on `/api/zones` and `POST /api/rescore`):**
   - *File:* `backend/api/schemas.py` lines 28–29
   - *Issue:* `ZoneProperties` defines `population: int` and `people_affected: int`, but `exposure_join.py` generates floating-point values (e.g., `36.99`, `7.10`).
   - *Impact:* Breaks the primary `/api/zones` endpoint, zone detail lookup `/api/zones/{id}`, and live rescoring endpoint `POST /api/rescore`. Causes 4 out of 5 `pytest` failures.
   - *Remedy:* Change both types to `float` (or round to `int` during zone generation / schema coercion).

2. **`score.py` Argument Parser Missing `--force`:**
   - *File:* `backend/priority/score.py`
   - *Issue:* When `python -m backend.run_pipeline --force` executes, it forwards `--force` to `backend/priority/score.py`, which rejects it (`unrecognized arguments: --force`).
   - *Impact:* The automated pipeline crashes before reaching `render` and `report` stages.
   - *Remedy:* Add `--force` argument to `backend/priority/score.py` or filter arguments passed from `run_pipeline.py`.

3. **Empty ML Output Rasters (`flood_ml.tif`, `water_pre.tif`, `water_post.tif`):**
   - *File:* `outputs/flood_ml.tif`, `backend/ml/`
   - *Issue:* All pixel values are `0.000` (min=0, max=0, mean=0). No pipeline CLI entry point (`infer.py`/`__main__.py`) exists in `backend/ml/`.
   - *Impact:* Machine learning is not contributing to flood detection. The fused output is 100% classical detection.

4. **Permanent Water Mask Inversion / Omission in Classical Detector:**
   - *File:* `backend/detect/classical.py`
   - *Issue:* All 3,698 flooded pixels in `flood_classical.tif` (1.48 km²) fall inside `perm_water.tif`. Permanent riverbed water of the Brahmaputra was flagged as flood.
   - *Impact:* False positives along the river channel rather than inundation on dry land.

---

### SHOULD FIX (Quality, Test, and Documentation improvements)
1. **Pytest Assertion Mismatch in `test_meta`:**
   - *File:* `tests/test_api.py` line 22
   - *Issue:* Test asserts `meta.get("demo") is True`, but when live/real data exists, `meta.json` specifies `"demo": false`.
   - *Remedy:* Check `assert "demo" in meta` or test against test fixtures.

2. **WeasyPrint Windows Setup in `README.md`:**
   - *Issue:* `pip install -r requirements-report.txt` installs WeasyPrint wheel, but execution requires native GTK+/Pango libraries on Windows.
   - *Remedy:* Document that Windows users must install GTK3 runtime or rely on HTML report fallback.

3. **Metadata Inconsistencies in `meta.json`:**
   - *Issue:* `s2_cloud_fractions` and `ml_available_fraction` are serialized as `null` instead of actual float values.

4. **AOI Reporting Ambiguity in `report.html`:**
   - *Issue:* The report header states `Total AOI area: 41.0 km²` (summing 41 priority grid cells of 1 km² each) instead of the actual geographic bounding box extent (897 km²).

---

### SKIP (Acceptable / By Design)
1. **Sentinel-1 Radiometric Outliers (0.03% - 0.07%):**
   - Extreme radar backscatter spikes outside [-40, +10 dB] are normal specular reflection / corner reflectors and do not impair thresholding.
2. **`flood_naive.tif` Overlap with Permanent Water:**
   - The naive detector is intentionally unmasked for baseline comparison purposes.
3. **High Confidence Zone Absence:**
   - Because ML produced zero flood detections, no pixel had consensus between classical and ML models. `Medium` and `Low` tiers are mathematically consistent with single-model agreement.

---

## 3. Honesty Audit & Evaluation

1. **Can this pipeline run end-to-end on a clean machine?**  
   *No.* Earth Engine authentication or pre-ingested assets are required. Running with `--force` terminates at the priority stage due to argument parsing errors.

2. **Is ML actually contributing to detection?**  
   *No.* ML rasters are 100% zero-valued. The pipeline ML step is bypassed because no entry point script exists. Classical thresholding accounts for 100% of detection.

3. **Does the API work with pipeline outputs?**  
   *Partially.* Readouts for `/api/health`, `/api/meta`, `/api/flood`, and `/api/overlays` succeed. `/api/zones` and live rescoring fail with 500 Internal Server Error due to float-to-int validation mismatch.

4. **Is permanent water properly excluded?**  
   *No.* The 3,698 flood pixels reside within the permanent water mask of the Brahmaputra River.

5. **Are impact figures sound and clearly caveated?**  
   *Yes, with caveats.* The reports and schemas explicitly label population estimates as model-derived estimates rather than empirical census counts. However, reporting 41 km² as the total AOI needs distinction between bounding box area and prioritized grid cell area.
