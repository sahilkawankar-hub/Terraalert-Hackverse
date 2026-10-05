"""
TerraAlert — Incident report generator.

Reads outputs/zones.geojson, outputs/meta.json and optional raster outputs to
produce a human-readable HTML report (and optionally PDF via WeasyPrint).

All numbers are computed from data.  No LLM-generated figures.

CLI:
    python -m backend.report.report [--force] [--no-pdf]
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger
from backend.common.meta import load_meta, update_meta

logger = get_logger("report")

TEMPLATE_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Data collection from outputs
# ---------------------------------------------------------------------------

def _collect_report_data() -> dict[str, Any]:
    """Read zones.geojson and meta.json, compute all report numbers."""
    meta_path = config.OUTPUTS_DIR / "meta.json"
    zones_path = config.OUTPUTS_DIR / "zones.geojson"

    # Fall back to demo data
    if not zones_path.exists():
        zones_path = config.OUTPUTS_DIR / "demo" / "zones.geojson"
    if not meta_path.exists():
        meta_path = config.OUTPUTS_DIR / "demo" / "meta.json"

    if not zones_path.exists():
        raise FileNotFoundError(
            f"zones.geojson not found in {config.OUTPUTS_DIR} or demo/. "
            "Run the zones stage first."
        )

    meta = load_meta(meta_path)
    with open(zones_path, "r", encoding="utf-8") as f:
        zones_fc = json.load(f)

    features = zones_fc.get("features", [])

    # Extract zone properties
    zones: list[dict[str, Any]] = []
    for feat in features:
        props = feat.get("properties", {})
        zones.append(props)

    # Sort by rank
    zones.sort(key=lambda z: z.get("rank", 9999))

    # Aggregate metrics
    total_flood_km2 = sum(z.get("flood_km2", 0) for z in zones)
    total_people = sum(z.get("people_affected", 0) for z in zones)
    total_facilities = sum(z.get("facilities_hit", 0) for z in zones)
    total_road_km = sum(z.get("road_cut_km", 0) for z in zones)

    # VERIFY zones
    verify_zones = [z for z in zones if z.get("tier") == "VERIFY"]

    # Confidence summary
    conf_counts: dict[str, int] = {}
    for z in zones:
        c = z.get("confidence", "Unknown")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    n_total = len(zones) or 1
    confidence_summary = [
        {"level": level, "count": count, "pct": count / n_total * 100}
        for level, count in sorted(conf_counts.items())
    ]

    # ML status
    fallbacks = meta.get("fallbacks", {})
    ml_status = fallbacks.get("ml", None)
    if ml_status:
        ml_model_status = f"Not run ({ml_status})"
    else:
        model = meta.get("model_name", config.ML_MODEL)
        ml_model_status = f"{model}"

    # Accuracy from meta
    accuracy_table = meta.get("metrics", None)
    if isinstance(accuracy_table, dict) and len(accuracy_table) > 0:
        # Round values
        accuracy_table = {k: round(v, 4) if isinstance(v, float) else v for k, v in accuracy_table.items()}
    else:
        accuracy_table = None

    # Trap comparison image
    trap_path = config.OUTPUTS_DIR / "trap_comparison.png"
    trap_exists = trap_path.exists()

    # Total AOI area in km² (sum of all grid cells)
    cell_km2 = (config.ZONE_CELL_M / 1000.0) ** 2
    total_aoi_km2 = round(len(zones) * cell_km2, 2)
    if total_aoi_km2 == 0 and len(config.AOI_BBOX) == 4:
        # Approximate from bbox if no zones: 1 deg lat ~ 111 km, lon ~ 111*cos(lat)
        min_lon, min_lat, max_lon, max_lat = config.AOI_BBOX
        mid_lat = math.radians((min_lat + max_lat) / 2)
        total_aoi_km2 = round(abs(max_lon - min_lon) * 111.0 * math.cos(mid_lat) * abs(max_lat - min_lat) * 111.0, 2)

    thresholds = meta.get("thresholds", {})

    return {
        "event_name": meta.get("event", config.EVENT_NAME),
        "aoi_bbox": meta.get("aoi_bbox", list(config.AOI_BBOX)),
        "total_aoi_km2": total_aoi_km2,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "satellite": meta.get("satellite", "Sentinel-1"),
        "pass_dir": meta.get("pass", config.S1_PASS),
        "relative_orbit": meta.get("relative_orbit", config.S1_RELATIVE_ORBIT),
        "pre_dates": meta.get("pre_dates", []),
        "post_dates": meta.get("post_dates", []),
        "time_gap_days": meta.get("time_gap_days", "N/A"),
        "total_flood_km2": round(total_flood_km2, 3),
        "total_people_affected": total_people,
        "total_facilities_hit": total_facilities,
        "total_road_cut_km": round(total_road_km, 2),
        "n_zones": len(zones),
        "top_3_zones": zones[:3],
        "top_zones": zones[:10],
        "verify_zones": verify_zones,
        "confidence_summary": confidence_summary,
        "post_db_max": thresholds.get("post_db_max", config.POST_DB_MAX),
        "diff_db_max": thresholds.get("diff_db_max", config.DIFF_DB_MAX),
        "slope_max_deg": thresholds.get("slope_max_deg", config.SLOPE_MAX_DEG),
        "min_obj_px": thresholds.get("min_object_pixels", config.MIN_OBJECT_PIXELS),
        "ml_model_status": ml_model_status,
        "fusion_rule": meta.get("fusion_rule", config.FUSION_RULE),
        "accuracy_table": accuracy_table,
        "trap_comparison_exists": trap_exists,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(force: bool = False, skip_pdf: bool = False) -> Path:
    """Generate the HTML incident report, and PDF if WeasyPrint is available.

    Returns the path to the primary report file (PDF if generated, else HTML).
    """
    html_path = config.OUTPUTS_DIR / "report.html"
    pdf_path = config.OUTPUTS_DIR / "report.pdf"

    if html_path.exists() and not force:
        logger.info("report.html exists. Use --force to regenerate.")
        return html_path

    # Collect data
    data = _collect_report_data()

    # Render Jinja2
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError(
            "jinja2 is required for report generation. "
            "Install with: pip install jinja2"
        )

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")
    html_content = template.render(**data)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info("Wrote %s", html_path)

    # Try PDF via WeasyPrint
    pdf_generated = False
    if not skip_pdf:
        try:
            from weasyprint import HTML
            HTML(string=html_content, base_url=str(config.OUTPUTS_DIR)).write_pdf(pdf_path)
            logger.info("Wrote %s", pdf_path)
            pdf_generated = True
        except (ImportError, OSError) as exc:
            logger.warning(
                "WeasyPrint or system graphics libraries (Pango/Cairo) not available (%s) — PDF skipped; HTML report retained.",
                exc,
            )
        except Exception as exc:
            logger.warning("WeasyPrint PDF generation failed: %s — keeping HTML only", exc)

    update_meta(
        config.OUTPUTS_DIR / "meta.json",
        "report",
        {
            "html": True,
            "pdf": pdf_generated,
            "generated_at": data["generated_at"],
        },
    )

    return pdf_path if pdf_generated else html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate TerraAlert incident report")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation")
    args = parser.parse_args()

    path = generate_report(force=args.force, skip_pdf=args.no_pdf)
    print(f"\n[DONE] Report -> {path}")


if __name__ == "__main__":
    main()
