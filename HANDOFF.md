# TerraAlert — Team Handoff Register

This document tracks cross-team handoffs between **Person A** (AI + detection) and **Person B** (zones + product).
Both engineers append one row each time they complete a deliverable that the other engineer depends on, or request an artifact/interface change.

| Date | From | To | What is ready or needed | File paths |
| :--- | :--- | :--- | :--- | :--- |
| 2026-10-06 | Shared | All | Working mode switched to Two-Engineer Mode; contract defined in AGENT_CONTEXT.md | `AGENT_CONTEXT.md`, `HANDOFF.md`, `config.py` |
| 2026-10-06 | Shared | All | Core OSM exposure and SAR ingestion pipelines established | `backend/ingest/exposure.py`, `data/roads.gpkg`, `data/facilities.gpkg` |
| 2026-10-06 | Shared | All | Atomic metadata merge helper `update_meta()` available for pipeline stages | `backend/common/meta.py` |
| 2026-10-06 | Person B | Person A / All | Grid generation (`build_zones.py`), exposure join (`exposure_join.py`), and priority scoring (`score.py`) implemented; 22 unit & integration tests passing. Ready to run on real data once `outputs/flood_fused.tif` or `outputs/flood_classical.tif` is produced by Person A. | `backend/zones/build_zones.py`, `backend/zones/exposure_join.py`, `backend/priority/score.py`, `tests/test_zones.py` |
