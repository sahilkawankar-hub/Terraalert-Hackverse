"""
Pydantic schemas and response models for the TerraAlert REST API.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "TerraAlert API"
    version: str = "0.1.0"


class Breakdown(BaseModel):
    severity: float
    people: float
    facilities: float
    roads: float


class ZoneProperties(BaseModel):
    zone_id: str
    name: str
    flood_pct: float
    flood_km2: float
    population: float
    people_affected: float
    facilities_hit: int
    road_cut_km: float
    confidence: str
    confidence_score: float
    reason: str
    facilities_by_type: dict[str, int] = Field(default_factory=dict)
    breakdown: Breakdown
    score: float
    rank: int
    tier: str
    zone_area_km2: Optional[float] = None
    conf_detail: Optional[dict[str, Any]] = None


class ZoneFeature(BaseModel):
    type: str = "Feature"
    id: Optional[str] = None
    geometry: dict[str, Any]
    properties: ZoneProperties


class ZoneCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[ZoneFeature]


class RescoreRequest(BaseModel):
    severity: float = Field(..., ge=0.0, description="Weight for flood severity factor (>= 0)")
    people: float = Field(..., ge=0.0, description="Weight for people affected factor (>= 0)")
    facilities: float = Field(..., ge=0.0, description="Weight for critical facilities factor (>= 0)")
    roads: float = Field(..., ge=0.0, description="Weight for cut road network factor (>= 0)")

    @model_validator(mode="after")
    def check_positive_sum(self) -> RescoreRequest:
        total = self.severity + self.people + self.facilities + self.roads
        if total <= 0:
            raise ValueError("At least one weight must be greater than 0; total sum cannot be 0")
        return self


class OverlaysResponse(BaseModel):
    bounds: list[float] = Field(..., description="[west, south, east, north] in EPSG:4326")
    corners: Optional[list[list[float]]] = Field(None, description="Four corner coordinates for MapLibre: [top-left, top-right, bottom-right, bottom-left]")
    note: str = "images not generated yet"
    layers: dict[str, str] = Field(default_factory=dict)
    pre: Optional[str] = None
    post: Optional[str] = None
    flood: Optional[str] = None
    confidence: Optional[str] = None


class ErrorDetail(BaseModel):
    detail: str
