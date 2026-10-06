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
| 2026-10-06 | Person A | Person B | ML flood detection pipeline COMPLETE. `flood_ml.tif`, `water_pre.tif`, and `water_post.tif` produced via Prithvi-EO 2.0 300M model. Spatial agreement with classical SAR in clear areas: **98.88%** (325,572 / 329,270 pixels). Grid alignment verified. | `outputs/flood_ml.tif`, `outputs/water_pre.tif`, `outputs/water_post.tif`, `backend/ml/temporal.py` |
| 2026-10-06 | Person A | Person B | Sentinel-2 ingest ready in `gee.py` (`--only s2`). Downloaded `s2_pre.tif` (97% clear) and `s2_post.tif` (30% clear). Cloud fractions recorded in `meta.json`. | `backend/ingest/gee.py`, `outputs/s2_pre.tif`, `outputs/s2_post.tif` |
