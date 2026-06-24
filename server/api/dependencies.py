"""Shared service container for FastAPI routes."""

import logging
from typing import Any, Dict

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ServiceContainer:
    def __init__(self):
        self.anonymization_service = None
        self.content_filter = None
        self.detector_manager = None
        self.config: Dict[str, Any] = {}

    def initialize(self, config: Dict[str, Any]):
        self.config = config
        logger.info("ServiceContainer initialized")

    def get_anonymization_service(self):
        if self.anonymization_service is None:
            raise HTTPException(503, "Anonymization service not initialized")
        return self.anonymization_service

    def get_content_filter(self):
        if self.content_filter is None:
            raise HTTPException(503, "Content filter not initialized")
        return self.content_filter

    def get_detector_manager(self):
        if self.detector_manager is None:
            raise HTTPException(503, "Detector manager not initialized")
        return self.detector_manager


services = ServiceContainer()
