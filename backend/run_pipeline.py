"""
TerraAlert — Full pipeline orchestrator.

Stages: ingest → detect → ml → fuse → zones → priority → render → report

Options:
    --stage <name>       Run only this stage
    --from-stage <name>  Run from this stage onward
    --force              Rebuild even if outputs exist
    --skip-ml            Skip the ML inference stage

Per-stage timing is reported.  Stages whose code does not exist yet (ml, fuse)
are skipped with a warning.  Person A's modules are invoked via their CLIs;
Person B never edits Person A's code.

CLI:
    python -m backend.run_pipeline
    python -m backend.run_pipeline --from-stage zones --force
    python -m backend.run_pipeline --stage report --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from backend.common.log import get_logger

logger = get_logger("pipeline")

# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "ingest",
    "detect",
    "ml",
    "fuse",
    "zones",
    "priority",
    "render",
    "report",
]


def _run_subprocess(cmd: list[str], label: str) -> bool:
    """Run a subprocess and log result.  Returns True if successful."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Stage '%s' failed with exit code %d", label, result.returncode)
            return False
        return True
    except FileNotFoundError:
        logger.error("Command not found: %s", cmd[0])
        return False


def _stage_ingest(force: bool) -> bool:
    """Stage 1: Data ingestion (Person A's gee.py + Person B's exposure.py)."""
    # Check if Person A's ingest exists
    gee_module = ROOT_DIR / "backend" / "ingest" / "gee.py"
    exposure_module = ROOT_DIR / "backend" / "ingest" / "exposure.py"

    success = True

    if gee_module.exists():
        args = [sys.executable, "-m", "backend.ingest.gee"]
        if force:
            args.append("--force")
        success = _run_subprocess(args, "ingest/gee")
    else:
        logger.warning("backend/ingest/gee.py not found — skipping SAR ingestion")

    if exposure_module.exists():
        args = [sys.executable, "-m", "backend.ingest.exposure"]
        if force:
            args.append("--force")
        ok = _run_subprocess(args, "ingest/exposure")
        success = success and ok
    else:
        logger.warning("backend/ingest/exposure.py not found — skipping exposure ingestion")

    return success


def _stage_detect(force: bool) -> bool:
    """Stage 2: Classical flood detection (Person A's module)."""
    detect_module = ROOT_DIR / "backend" / "detect"
    # Look for the main detection entry point
    candidates = [
        detect_module / "classical.py",
        detect_module / "detect.py",
        detect_module / "__main__.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            args = [sys.executable, "-m", "backend.detect.classical"]
            if candidate.name == "detect.py":
                args = [sys.executable, "-m", "backend.detect.detect"]
            elif candidate.name == "__main__.py":
                args = [sys.executable, "-m", "backend.detect"]
            if force:
                args.append("--force")
            return _run_subprocess(args, "detect")

    # Check required inputs
    pre_path = config.OUTPUTS_DIR / "pre.tif"
    post_path = config.OUTPUTS_DIR / "post.tif"
    if not pre_path.exists() or not post_path.exists():
        logger.warning(
            "Detection stage skipped: requires outputs/pre.tif and outputs/post.tif "
            "(run ingest first, or Person A must produce these)"
        )
    else:
        logger.warning("Detection module not found in backend/detect/ — skipping")

    return True  # Not a hard failure if Person A hasn't built it yet


def _stage_ml(force: bool, skip_ml: bool) -> bool:
    """Stage 3: ML inference (Person A's module — optional)."""
    if skip_ml:
        logger.info("ML stage skipped (--skip-ml)")
        return True

    ml_module = ROOT_DIR / "backend" / "ml"
    candidates = [
        ml_module / "temporal.py",
        ml_module / "infer.py",
        ml_module / "predict.py",
        ml_module / "__main__.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            mod_name = f"backend.ml.{candidate.stem}" if candidate.name != "__main__.py" else "backend.ml"
            args = [sys.executable, "-m", mod_name]
            if force:
                args.append("--force")
            return _run_subprocess(args, "ml")

    logger.warning("ML module not found in backend/ml/ — skipping (Person A)")
    return True


def _stage_fuse(force: bool) -> bool:
    """Stage 4: Fusion (Person A's module)."""
    fusion_module = ROOT_DIR / "backend" / "fusion"
    candidates = [
        fusion_module / "fuse.py",
        fusion_module / "fusion.py",
        fusion_module / "__main__.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            mod_name = f"backend.fusion.{candidate.stem}" if candidate.name != "__main__.py" else "backend.fusion"
            args = [sys.executable, "-m", mod_name]
            if force:
                args.append("--force")
            return _run_subprocess(args, "fuse")

    logger.warning("Fusion module not found in backend/fusion/ — skipping (Person A)")
    return True


def _stage_zones(force: bool) -> bool:
    """Stage 5: Zone grid + exposure join."""
    args = [sys.executable, "-m", "backend.zones.build_zones"]
    if force:
        args.append("--force")
    return _run_subprocess(args, "zones")


def _stage_priority(force: bool) -> bool:
    """Stage 6: Priority scoring (re-score in place)."""
    zones_path = config.OUTPUTS_DIR / "zones.geojson"
    if not zones_path.exists():
        demo_zones = config.OUTPUTS_DIR / "demo" / "zones.geojson"
        if not demo_zones.exists():
            logger.error(
                "Priority stage requires outputs/zones.geojson — run zones stage first"
            )
            return False
        logger.info("Using demo zones for priority scoring")
    # Priority is already embedded in the zones stage; this is for standalone rescoring
    args = [sys.executable, "-m", "backend.priority.score"]
    if force:
        args.append("--force")
    # The score module may not have --force; just run it and tolerate failure
    try:
        return _run_subprocess(args, "priority")
    except Exception:
        logger.info("Priority re-score skipped (already applied during zones stage)")
        return True


def _stage_render(force: bool) -> bool:
    """Stage 7: Render raster overlays."""
    args = [sys.executable, "-m", "backend.api.render"]
    if force:
        args.append("--force")
    return _run_subprocess(args, "render")


def _stage_report(force: bool) -> bool:
    """Stage 8: Generate HTML/PDF incident report."""
    args = [sys.executable, "-m", "backend.report.report"]
    if force:
        args.append("--force")
    return _run_subprocess(args, "report")


# Stage dispatcher
STAGE_FUNCS = {
    "ingest": lambda f, **kw: _stage_ingest(f),
    "detect": lambda f, **kw: _stage_detect(f),
    "ml": lambda f, **kw: _stage_ml(f, kw.get("skip_ml", False)),
    "fuse": lambda f, **kw: _stage_fuse(f),
    "zones": lambda f, **kw: _stage_zones(f),
    "priority": lambda f, **kw: _stage_priority(f),
    "render": lambda f, **kw: _stage_render(f),
    "report": lambda f, **kw: _stage_report(f),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(
    stage: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    skip_ml: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline or a subset of stages.

    Returns a dict mapping stage name → {"ok": bool, "seconds": float}.
    """
    if stage:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}. Choose from {STAGE_ORDER}")
        stages_to_run = [stage]
    elif from_stage:
        if from_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {from_stage}. Choose from {STAGE_ORDER}")
        idx = STAGE_ORDER.index(from_stage)
        stages_to_run = STAGE_ORDER[idx:]
    else:
        stages_to_run = STAGE_ORDER[:]

    results: dict[str, dict[str, Any]] = {}

    logger.info("=" * 60)
    logger.info("TerraAlert Pipeline — stages: %s", ", ".join(stages_to_run))
    logger.info("=" * 60)

    pipeline_start = time.time()

    for stage_name in stages_to_run:
        logger.info("── Stage: %s ──", stage_name)
        t0 = time.time()

        func = STAGE_FUNCS[stage_name]
        try:
            ok = func(force, skip_ml=skip_ml)
        except Exception as exc:
            logger.error("Stage '%s' raised exception: %s", stage_name, exc)
            ok = False

        elapsed = time.time() - t0
        results[stage_name] = {"ok": ok, "seconds": round(elapsed, 2)}
        status_str = "[OK]" if ok else "[FAIL]"
        logger.info("  %s %s (%.2fs)", status_str, stage_name, elapsed)

        if not ok:
            logger.error("Pipeline stopped at stage '%s'", stage_name)
            break

    total = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1fs", total)
    for s, r in results.items():
        mark = "[OK]" if r["ok"] else "[FAIL]"
        logger.info("  %-6s %-12s %6.2fs", mark, s, r["seconds"])
    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TerraAlert pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Stages: {', '.join(STAGE_ORDER)}",
    )
    parser.add_argument("--stage", type=str, help="Run only this stage")
    parser.add_argument("--from-stage", type=str, help="Run from this stage onward")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML inference stage")
    args = parser.parse_args()

    results = run_pipeline(
        stage=args.stage,
        from_stage=args.from_stage,
        force=args.force,
        skip_ml=args.skip_ml,
    )

    # Exit with failure if any stage failed
    if any(not r["ok"] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
