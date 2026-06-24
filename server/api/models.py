"""Pydantic response models for the anonymization API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float
    mask: Optional[str] = Field(None, description="base64-encoded PNG mask")


class AnonymizationResponse(BaseModel):
    success: bool
    anonymized_image: Optional[str] = None
    original_image: Optional[str] = None
    detections: Optional[Dict[str, List[DetectionResult]]] = None
    processing_time_ms: float
    method_used: Dict[str, str]
    warnings: List[str] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    sd_api_available: bool
    gpu_count: int
    content_filter_loaded: bool
    detectors_loaded: bool
    message: str


class MethodsResponse(BaseModel):
    targets: Dict[str, List[str]]
    detectors: List[str]
