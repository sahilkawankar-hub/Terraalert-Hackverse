# 🌊 TerraAlert — Post-Disaster Flood Intelligence & Priority Ranking

> **Rapid, multi-sensor satellite change detection, population exposure estimation, and infrastructure impact scoring for emergency disaster response.**

TerraAlert provides operational emergency management teams with rapid, automated situational intelligence following flood disasters. By combining multi-temporal Sentinel-1 Synthetic Aperture Radar (SAR) imagery, OpenStreetMap critical infrastructure topologies, and WorldPop high-resolution gridded demographics, TerraAlert divides the disaster zone into actionable operational grid cells, estimates compounding humanitarian risk, and outputs ranked priority zones for rescue deployment.

---

## 🏗️ Architecture & Two-Engineer Division of Labor

TerraAlert is architected with strict decoupling between remote sensing detection modules and spatial exposure/product delivery:

```
                          ┌───────────────────────────┐
                          │   Sentinel-1 SAR / GEE    │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌─────────────────────────────────┐       ┌───────────────────────────────────┐
│     PERSON A (AI & Detection)   │       │   PERSON B (Zones & Product)      │
│  - backend/detect/              │       │  - backend/zones/                 │
│    (Classical VV change diff)   │       │    (AOI grid, flood vectorization,│
│  - backend/ml/                  │       │     zonal exposure aggregation)   │
│    (Deep learning segmentation) │       │  - backend/priority/              │
│  - backend/fusion/              │       │    (Multi-factor scoring, rescore)│
│    (Multi-sensor fusion)        │       │  - backend/report/                │
│  - backend/metrics/             │       │    (HTML/PDF incident reporting)  │
│    (Sen1Floods11 validation)    │       │  - backend/api/                   │
│                                 │       │    (FastAPI endpoints & overlays) │
│                                 │       │  - backend/run_pipeline.py        │
│                                 │       │  - scripts/build_demo_bundle.py   │
└────────────────┬────────────────┘       └─────────────────┬─────────────────┘
                 │                                          │
                 │ outputs/flood_*.tif                      │ outputs/zones.geojson
                 └────────────────────► ◄───────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   FastAPI Web Platform &    │
                         │   MapLibre GL Frontend      │
                         └─────────────────────────────┘
```

### Team Responsibilities
- **Person A (AI & Detection):** Owns `backend/detect/`, `backend/ml/`, `backend/fusion/`, `backend/metrics/`. Responsible for raw SAR/optical ingestion, decibel backscatter differencing, neural net segmentation, permanent water masking, and model accuracy benchmarking.
- **Person B (Zones & Product):** Owns `backend/zones/`, `backend/priority/`, `backend/report/`, `backend/api/`, `backend/run_pipeline.py`, `scripts/build_demo_bundle.py`, and project documentation (`README.md`, `SOURCES.md`, `LIMITATIONS.md`). Responsible for grid generation, exposure joins (roads, facilities, population), dynamic multi-factor scoring, transparent map overlays, and automated reporting.

---

## ⚡ Quickstart

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/YourOrg/Terraalert-Hackverse.git
cd Terraalert-Hackverse
pip install -r requirements.txt
pip install -r requirements-report.txt
```

### 2. Standalone Demo Mode (Zero Dependencies / Instant Run)
TerraAlert includes a committed, lightweight demo bundle under `outputs/demo/` requiring no external credentials or satellite downloads:
```bash
# Generate mock demo data if starting fresh
python scripts/make_mock_demo.py

# Package demo bundle
python scripts/build_demo_bundle.py

# Launch the FastAPI server with embedded frontend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to:
- **Web UI & Interactive Map:** `http://localhost:8000/`
- **Interactive Swagger API Docs:** `http://localhost:8000/docs`
- **Direct HTML Incident Report:** `http://localhost:8000/api/report`

---

## 🚀 Running the Full Pipeline

The automated orchestrator `backend.run_pipeline` manages stage execution, error isolation, and execution timing:

```bash
# Run all available pipeline stages
python -m backend.run_pipeline

# Run from zones stage onwards (skipping already-generated SAR inputs)
python -m backend.run_pipeline --from-stage zones --force

# Run only a specific stage
python -m backend.run_pipeline --stage render --force
python -m backend.run_pipeline --stage report --force
```

### Pipeline Stages
1. **`ingest`**: Downloads Sentinel-1 SAR tiles and fetches OSM road/facility vectors.
2. **`detect`**: Performs multi-temporal change detection ($\Delta\sigma^0 \le -3\text{ dB}$, $\text{slope} \le 5^\circ$, JRC permanent water mask).
3. **`ml`**: (Optional) Runs deep learning segmentation on SAR/optical chips.
4. **`fuse`**: Combines rule-based and ML detections into a fused flood probability map.
5. **`zones`**: Generates uniform metric grid cells (default $1\,000\text{ m} \times 1\,000\text{ m}$ in `GRID_CRS`), vectorizes flood masks, and calculates zonal exposure (population, cut roads, facilities hit).
6. **`priority`**: Normalizes exposure factors, computes compound risk scores, ranks zones, and assigns tiers (`P1`, `P2`, `P3`, `VERIFY`).
7. **`render`**: Converts spatial GeoTIFFs to transparent PNG overlays and writes MapLibre `bounds.json`.
8. **`report`**: Compiles an incident report (HTML and PDF) containing all calculated metrics.

---

## 🎯 Priority Scoring & The `VERIFY` Protocol

TerraAlert evaluates compounding vulnerability per zone using four normalized factors:

$$\text{Score} = w_{\text{sev}} \cdot S_{\text{severity}} + w_{\text{pop}} \cdot S_{\text{people}} + w_{\text{fac}} \cdot S_{\text{facilities}} + w_{\text{roads}} \cdot S_{\text{roads}}$$

- **Severity ($S_{\text{severity}}$):** Flooded area fraction of the zone ($0.0 - 1.0$).
- **People Affected ($S_{\text{people}}$):** Estimated exposed population (log-normalized).
- **Critical Facilities ($S_{\text{facilities}}$):** Inundated hospitals, clinics, schools, and emergency stations.
- **Cut Roads ($S_{\text{roads}}$):** Length of submerged/impassable road segments ($\text{km}$).

### Priority Tiers
- **P1 (Critical):** Score $\ge 0.70$ (top-priority rescue and staging zones).
- **P2 (High):** $0.40 \le \text{Score} < 0.70$.
- **P3 (Moderate):** Score $< 0.40$.
- **VERIFY (Operational Safeguard):** Any zone qualifying for P1 or P2 whose detection confidence is tagged **Low** (due to steep slope, multi-sensor disagreement, or radar shadow). **Emergency crews must verify on-ground before deploying heavy assets.**

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Operational health and service metadata. |
| `/api/meta` | `GET` | Event parameters, bounding box, sensor orbits, thresholds, and fallback flags. |
| `/api/zones` | `GET` | Ranked priority zones GeoJSON. Supports query filters: `?tier=P1,P2` and `?min_confidence=High`. |
| `/api/zones/{zone_id}` | `GET` | Detailed properties and breakdown for an individual zone. |
| `/api/rescore` | `POST` | Re-score all zones dynamically based on custom factor weights (`severity`, `people`, `facilities`, `roads`). |
| `/api/flood` | `GET` | Detected flood boundary polygons as GeoJSON. |
| `/api/facilities` | `GET` | Critical facilities (hospitals, schools, fire) with `in_flood` boolean flags. |
| `/api/roads-cut` | `GET` | Inundated and impassable road line segments. |
| `/api/overlays` | `GET` | Raster overlay URLs (`pre`, `post`, `flood`, `confidence`) and MapLibre corner bounds. |
| `/api/overlays/{filename}` | `GET` | Serves static transparent PNG overlays or `bounds.json`. |
| `/api/report` | `GET` | Serves the generated incident report (`?format=html` or `?format=pdf`). |

---

## 🧪 Testing & Validation

The test suite validates algorithmic contracts, coordinate reference system alignment, and API integrity without network access:

```bash
# Run unit & integration tests
pytest tests/test_zones.py tests/test_meta.py tests/test_grid.py tests/test_api.py tests/test_exposure.py

# Run all tests
pytest
```

---

## 📚 Documentation & Reference

- [Data Sources & Citations (`SOURCES.md`)](file:///c:/Users/Admin/OneDrive/Desktop/TerraAlert/Terraalert-Hackverse/SOURCES.md)
- [Physical Limitations, Sensor Physics & Caveats (`LIMITATIONS.md`)](file:///c:/Users/Admin/OneDrive/Desktop/TerraAlert/Terraalert-Hackverse/LIMITATIONS.md)
- [Inter-Team Handoff Register (`HANDOFF.md`)](file:///c:/Users/Admin/OneDrive/Desktop/TerraAlert/Terraalert-Hackverse/HANDOFF.md)