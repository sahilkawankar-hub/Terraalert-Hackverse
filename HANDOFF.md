# TerraAlert — Team Handoff Register

This document tracks cross-team handoffs between **Person A** (AI + detection) and **Person B** (zones + product).
Both engineers append one row each time they complete a deliverable that the other engineer depends on, or request an artifact/interface change.

| Date | From | To | What is ready or needed | File paths |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-06 | Shared | All | Working mode switched to Two-Engineer Mode; contract defined in AGENT_CONTEXT.md | `AGENT_CONTEXT.md`, `HANDOFF.md`, `config.py` |
| 2026-10-06 | Shared | All | Core OSM exposure and SAR ingestion pipelines established | `backend/ingest/exposure.py`, `data/roads.gpkg`, `data/facilities.gpkg` |
| 2026-10-06 | Shared | All | Atomic metadata merge helper `update_meta()` available for pipeline stages | `backend/common/meta.py` |
| 2026-10-06 | Person A | Person B | Classical SAR flood detection mask ready (1.48 km² flooded, main river excluded) | `outputs/flood_classical.tif`, `outputs/trap_comparison.png` |
| 2026-10-06 | Person A | Person B | `add_zone_confidence(zones, outputs_dir)` and `confidence_pixel.tif` ready | `backend/fusion/confidence.py`, `outputs/confidence_pixel.tif` |
| 2026-10-06 | Person A | Person B | ML pipeline ready. **`flood_ml.tif` requires GPU run via Colab** (torch not installed locally). Once `flood_ml.tif` is produced from `scripts/run_ml_colab.ipynb` and copied to `outputs/`, Person B's zone scoring can use it directly. Until then `flood_fused.tif` falls back to `flood_classical.tif` per contract. | `backend/ml/prithvi.py`, `backend/ml/temporal.py`, `scripts/run_ml_colab.ipynb` |
| 2026-10-06 | Person A | Person B | Sentinel-2 ingest added to `gee.py` (`--only s2`). Run `python -m backend.ingest.gee --only s2` to produce `s2_pre.tif` and `s2_post.tif` (needs GEE auth). Cloud fractions will be recorded in `meta.json` under `s2.*`. | `backend/ingest/gee.py`, `outputs/s2_pre.tif`, `outputs/s2_post.tif` |
