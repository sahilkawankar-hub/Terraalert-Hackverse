"""
TerraAlert — Priority scoring and zone ranking module.

Provides a pure function `rescore(zones, weights)` with a stable signature.
Given a collection of zone features and factor weights, normalises factor values,
computes weighted priority scores, updates rank and breakdown, and assigns tiers
(P1, P2, P3, or VERIFY for P1/P2 zones with Low confidence).
"""
from __future__ import annotations

import copy
import math
from typing import Any

import config


def rescore(
    zones: list[dict[str, Any]] | dict[str, Any],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Pure function to score, rank, and tier zones based on multi-factor weights.

    Parameters
    ----------
    zones : list of GeoJSON features or a GeoJSON FeatureCollection dict.
    weights : dict mapping factors ("severity", "people", "facilities", "roads")
              to non-negative float weights. Defaults to config.WEIGHTS if None.

    Returns
    -------
    The same type as input (FeatureCollection or list of features), with updated
    `score`, `rank`, `tier`, `breakdown`, and `reason` fields.
    """
    if weights is None:
        raw_weights = copy.deepcopy(config.WEIGHTS)
    else:
        raw_weights = dict(weights)

    # Validate weights
    factors = ["severity", "people", "facilities", "roads"]
    for f in factors:
        val = raw_weights.get(f, 0.0)
        if val < 0:
            raise ValueError(f"Weight for '{f}' cannot be negative: {val}")

    total_weight = sum(raw_weights.get(f, 0.0) for f in factors)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be strictly positive")

    # Normalise weights to sum to 1.0
    norm_weights = {f: raw_weights.get(f, 0.0) / total_weight for f in factors}

    # Handle FeatureCollection vs list of features
    is_feature_collection = isinstance(zones, dict) and zones.get("type") == "FeatureCollection"
    if is_feature_collection:
        feature_list = copy.deepcopy(zones.get("features", []))
    elif isinstance(zones, list):
        feature_list = copy.deepcopy(zones)
    elif isinstance(zones, dict) and "features" in zones:
        feature_list = copy.deepcopy(zones["features"])
        is_feature_collection = True
    else:
        raise TypeError("Expected a GeoJSON FeatureCollection dict or list of feature dicts")

    if not feature_list:
        if is_feature_collection:
            return {"type": "FeatureCollection", "features": []}
        return []

    # Extract properties or direct dictionaries
    extracted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for feat in feature_list:
        props = feat.get("properties") if isinstance(feat, dict) and "properties" in feat else feat
        extracted.append((feat, props))

    # Determine max values across all zones for min-max / proportional scaling
    max_severity = max((float(p.get("flood_pct", 0.0)) for _, p in extracted), default=0.0)
    max_people = max((float(p.get("people_affected", 0.0)) for _, p in extracted), default=0.0)
    max_facilities = max((float(p.get("facilities_hit", 0.0)) for _, p in extracted), default=0.0)
    max_roads = max((float(p.get("road_cut_km", 0.0)) for _, p in extracted), default=0.0)

    # Calculate raw scores and breakdown for each zone
    scored_items: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for feat, props in extracted:
        sev_val = float(props.get("flood_pct", 0.0))
        peop_val = float(props.get("people_affected", 0.0))
        fac_val = float(props.get("facilities_hit", 0.0))
        rd_val = float(props.get("road_cut_km", 0.0))

        # Proportional normalisation [0, 1] per factor
        s_norm = (sev_val / max_severity) if max_severity > 0 else 0.0
        p_norm = (peop_val / max_people) if max_people > 0 else 0.0
        f_norm = (fac_val / max_facilities) if max_facilities > 0 else 0.0
        r_norm = (rd_val / max_roads) if max_roads > 0 else 0.0

        c_sev = norm_weights["severity"] * s_norm
        c_peop = norm_weights["people"] * p_norm
        c_fac = norm_weights["facilities"] * f_norm
        c_rd = norm_weights["roads"] * r_norm

        total_score = c_sev + c_peop + c_fac + c_rd

        props["breakdown"] = {
            "severity": round(c_sev, 4),
            "people": round(c_peop, 4),
            "facilities": round(c_fac, 4),
            "roads": round(c_rd, 4),
        }
        props["score"] = round(total_score, 4)
        scored_items.append((total_score, feat, props))

    # Sort descending by score; tie-break on people_affected, facilities_hit, zone_id
    scored_items.sort(
        key=lambda item: (
            item[0],
            float(item[2].get("people_affected", 0.0)),
            float(item[2].get("facilities_hit", 0.0)),
            str(item[2].get("zone_id", "")),
        ),
        reverse=True,
    )

    n_zones = len(scored_items)
    n_p1 = max(1, math.ceil(n_zones * config.TIER_P1_FRAC))
    n_p2 = max(1, math.ceil(n_zones * config.TIER_P2_FRAC))

    ranked_features: list[dict[str, Any]] = []
    for rank_idx, (_, feat, props) in enumerate(scored_items, start=1):
        props["rank"] = rank_idx

        # Assign preliminary tier
        if rank_idx <= n_p1:
            raw_tier = "P1"
        elif rank_idx <= n_p1 + n_p2:
            raw_tier = "P2"
        else:
            raw_tier = "P3"

        # Check confidence for VERIFY rule:
        # P1 or P2 zones with Low confidence are tagged as VERIFY
        conf_label = str(props.get("confidence", "Medium")).strip().capitalize()
        if raw_tier in ("P1", "P2") and conf_label == "Low":
            props["tier"] = "VERIFY"
            props["reason"] = (
                f"Flagged for on-ground verification: High impact ({raw_tier}) "
                f"with Low detection confidence"
            )
        else:
            props["tier"] = raw_tier

        ranked_features.append(feat)

    if is_feature_collection:
        result = copy.deepcopy(zones)
        result["features"] = ranked_features
        return result

    return ranked_features
