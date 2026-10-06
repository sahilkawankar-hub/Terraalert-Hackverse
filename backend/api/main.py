"""
TerraAlert — REST API entrypoint.

FastAPI application providing endpoints for flood intelligence:
  - System health & metadata
  - Filterable priority zones (GeoJSON) and zone detail
  - Flood masks, critical facilities, cut roads (GeoJSON)
  - Interactive multi-factor weight rescoring
  - Satellite/analytical overlays
  - PDF/HTML incident report retrieval
Serves static frontend at "/".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from backend.api.schemas import (
    HealthResponse,
    OverlaysResponse,
    RescoreRequest,
    ZoneCollection,
    ZoneFeature,
)
from backend.priority.score import rescore

app = FastAPI(
    title="TerraAlert API",
    description="Post-disaster flood change detection, impact scoring, and priority zone ranking.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data file resolution helpers ─────────────────────────────────────────────
def _resolve_data_path(filename: str) -> tuple[Path, bool]:
    """Resolve a data file path, falling back to outputs/demo/ if real file is absent."""
    real_path = config.OUTPUTS_DIR / filename
    if real_path.exists():
        return real_path, False

    demo_path = config.OUTPUTS_DIR / "demo" / filename
    if demo_path.exists():
        return demo_path, True

    return real_path, False


def _load_geojson(filename: str, stage_hint: str) -> tuple[dict[str, Any], bool]:
    """Load GeoJSON file from outputs/ or outputs/demo/, or raise a clear 404/409."""
    path, is_demo = _resolve_data_path(filename)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{filename} not found. {stage_hint}",
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, is_demo
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read {filename}: {exc}",
        ) from exc


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Return system operational status and service information."""
    return HealthResponse()


@app.get("/api/meta", tags=["System"])
def get_meta() -> dict[str, Any]:
    """Return pipeline metadata, configuration, thresholds, and execution state."""
    path, is_demo = _resolve_data_path("meta.json")
    if not path.exists():
        # Fallback to dynamic metadata if no file exists yet
        return {
            "event": config.EVENT_NAME,
            "aoi_bbox": list(config.AOI_BBOX),
            "demo": True,
            "fallbacks": {"ml": "not run"},
            "note": "No meta.json generated yet. Run pipeline or make_mock_demo.py.",
        }

    with open(path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # If running on demo bundle or real zones do not exist, ensure demo=True
    zones_real = config.OUTPUTS_DIR / "zones.geojson"
    if is_demo or not zones_real.exists():
        meta_data["demo"] = True

    return meta_data


@app.get("/api/zones", response_model=ZoneCollection, tags=["Zones"])
def get_zones(
    min_confidence: Optional[str] = Query(
        None,
        description="Filter by minimum confidence: Low, Medium, High",
    ),
    tier: Optional[str] = Query(
        None,
        description="Comma-separated tier list to include: P1, P2, P3, VERIFY",
    ),
) -> ZoneCollection:
    """Return ranked priority zones as a GeoJSON FeatureCollection with optional filters."""
    data, _ = _load_geojson(
        "zones.geojson",
        stage_hint="Run stage 4 (zones/priority) or python scripts/make_mock_demo.py first.",
    )

    features = data.get("features", [])

    # Filter by confidence level
    conf_ranks = {"low": 1, "medium": 2, "high": 3}
    if min_confidence:
        min_c_clean = min_confidence.strip().lower()
        if min_c_clean not in conf_ranks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid min_confidence '{min_confidence}'. Expected Low, Medium, or High.",
            )
        target_rank = conf_ranks[min_c_clean]
        features = [
            f for f in features
            if conf_ranks.get(str(f.get("properties", {}).get("confidence", "")).lower(), 0) >= target_rank
        ]

    # Filter by priority tier
    if tier:
        allowed_tiers = {t.strip().upper() for t in tier.split(",") if t.strip()}
        features = [
            f for f in features
            if str(f.get("properties", {}).get("tier", "")).upper() in allowed_tiers
        ]

    return ZoneCollection(type="FeatureCollection", features=features)


@app.get("/api/zones/{zone_id}", response_model=ZoneFeature, tags=["Zones"])
def get_zone_by_id(zone_id: str) -> ZoneFeature:
    """Return a single priority zone by its identifier."""
    data, _ = _load_geojson(
        "zones.geojson",
        stage_hint="Run stage 4 (zones/priority) or python scripts/make_mock_demo.py first.",
    )

    clean_id = zone_id.strip().upper()
    for feat in data.get("features", []):
        curr_id = str(feat.get("properties", {}).get("zone_id", "")).strip().upper()
        if curr_id == clean_id:
            return ZoneFeature(**feat)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Zone '{zone_id}' not found.",
    )


@app.get("/api/flood", tags=["Geospatial Layers"])
def get_flood_mask() -> dict[str, Any]:
    """Return the detected flood extent polygons as GeoJSON."""
    data, _ = _load_geojson(
        "flood_mask.geojson",
        stage_hint="Run stage 1/2 (detection/fusion) or python scripts/make_mock_demo.py first.",
    )
    return data


@app.get("/api/facilities", tags=["Geospatial Layers"])
def get_facilities() -> dict[str, Any]:
    """Return critical infrastructure facilities (with inundation flag) as GeoJSON."""
    data, _ = _load_geojson(
        "facilities.geojson",
        stage_hint="Run stage 3 (exposure) or python scripts/make_mock_demo.py first.",
    )
    return data


@app.get("/api/roads-cut", tags=["Geospatial Layers"])
def get_roads_cut() -> dict[str, Any]:
    """Return impassable/submerged road segments as GeoJSON."""
    data, _ = _load_geojson(
        "roads_cut.geojson",
        stage_hint="Run stage 3 (exposure) or python scripts/make_mock_demo.py first.",
    )
    return data


@app.post("/api/rescore", response_model=ZoneCollection, tags=["Priority Scoring"])
def post_rescore(body: RescoreRequest) -> ZoneCollection:
    """Re-score and re-rank all zones dynamically based on user-adjusted factor weights.

    Normalises weights to sum to 1.0, updates score/rank/tier, and returns the
    re-ranked GeoJSON FeatureCollection.
    """
    data, _ = _load_geojson(
        "zones.geojson",
        stage_hint="Run stage 4 (zones/priority) or python scripts/make_mock_demo.py first.",
    )

    weights_dict = {
        "severity": body.severity,
        "people": body.people,
        "facilities": body.facilities,
        "roads": body.roads,
    }

    try:
        rescored_data = rescore(data, weights_dict)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return ZoneCollection(**rescored_data)


@app.get("/api/overlays", response_model=OverlaysResponse, tags=["Overlays"])
def get_overlays() -> OverlaysResponse:
    """Return raster overlay layer URLs and spatial bounds.

    Checks outputs/overlays/ (and fallback outputs/demo/overlays/) for bounds.json
    and rendered PNGs (pre, post, flood, confidence).
    """
    bounds = list(config.AOI_BBOX)
    corners: Optional[list[list[float]]] = None

    bounds_file, _ = _resolve_data_path(Path("overlays") / "bounds.json")
    if bounds_file.exists():
        try:
            with open(bounds_file, "r", encoding="utf-8") as f:
                bdata = json.load(f)
                bounds = bdata.get("bounds", bounds)
                corners = bdata.get("corners", None)
        except Exception:
            pass

    # Discover available layer PNGs
    layer_names = ["pre", "post", "flood", "confidence"]
    available_layers: dict[str, str] = {}
    missing_layers: list[str] = []

    for name in layer_names:
        png_path, _ = _resolve_data_path(Path("overlays") / f"{name}.png")
        if png_path.exists():
            available_layers[name] = f"/api/overlays/{name}.png"
        else:
            missing_layers.append(name)

    if not available_layers:
        note = "images not generated yet; run render stage or python -m backend.api.render"
    elif "confidence" not in available_layers:
        note = "confidence layer skipped (confidence_pixel.tif not generated yet)"
    else:
        note = "All raster overlay layers available"

    return OverlaysResponse(
        bounds=bounds,
        corners=corners,
        note=note,
        layers=available_layers,
        pre=available_layers.get("pre"),
        post=available_layers.get("post"),
        flood=available_layers.get("flood"),
        confidence=available_layers.get("confidence"),
    )


@app.get("/api/overlays/{filename}", tags=["Overlays"])
def get_overlay_file(filename: str) -> FileResponse:
    """Serve a rendered transparent PNG overlay or bounds.json."""
    real_path = config.OUTPUTS_DIR / "overlays" / filename
    demo_path = config.OUTPUTS_DIR / "demo" / "overlays" / filename
    target = real_path if real_path.exists() else demo_path

    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Overlay '{filename}' not found. Run python -m backend.api.render first.",
        )

    media_type = "application/json" if filename.endswith(".json") else "image/png"
    return FileResponse(target, media_type=media_type)


@app.get("/api/report", tags=["Reporting"])
def get_report(format: Optional[str] = Query(None, description="Preferred format: 'pdf' or 'html'")) -> FileResponse:
    """Retrieve the generated PDF or HTML incident report."""
    report_pdf = config.OUTPUTS_DIR / "report.pdf"
    demo_pdf = config.OUTPUTS_DIR / "demo" / "report.pdf"
    report_html = config.OUTPUTS_DIR / "report.html"
    demo_html = config.OUTPUTS_DIR / "demo" / "report.html"

    target_pdf = report_pdf if report_pdf.exists() else (demo_pdf if demo_pdf.exists() else None)
    target_html = report_html if report_html.exists() else (demo_html if demo_html.exists() else None)

    fmt = (format or "").strip().lower()

    if fmt == "pdf":
        if target_pdf:
            return FileResponse(target_pdf, media_type="application/pdf", filename="report.pdf")
        if target_html:
            return FileResponse(target_html, media_type="text/html", filename="report.html")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF report not found. Run python -m backend.report.report first.",
        )

    # Default to HTML
    if target_html:
        return FileResponse(target_html, media_type="text/html", filename="report.html")
    if target_pdf:
        return FileResponse(target_pdf, media_type="application/pdf", filename="report.pdf")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Incident report has not been generated yet. Run python -m backend.report.report first.",
    )


@app.post("/api/report/generate", tags=["Reporting"])
def generate_incident_report() -> dict[str, Any]:
    """Generate or re-generate both HTML and PDF incident reports on demand."""
    from backend.report.report import generate_report
    try:
        generate_report(force=True, skip_pdf=False)
        has_pdf = (config.OUTPUTS_DIR / "report.pdf").exists()
        return {
            "status": "success",
            "message": "Incident report generated successfully",
            "pdf_url": "/api/report?format=pdf",
            "html_url": "/api/report?format=html",
            "has_pdf": has_pdf,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate incident report: {exc}",
        )


# ── No-cache middleware for HTML (prevent stale browser cache) ─────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: StarletteResponse = await call_next(request)
        if request.url.path.endswith(".html") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheHTMLMiddleware)

# ── Mount Frontend Static Files at "/" ────────────────────────────────────────
FRONTEND_DIR = config.ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
