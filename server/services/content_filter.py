"""Content filtering service.

This is a no-op passthrough. A real NSFW/gore detector (AWS Rekognition,
Azure Content Moderator, a local NSFW model, ...) should replace this if you
need filtering for a public-facing deploy.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FilterResult:
    def __init__(self, is_safe: bool = True, reason: str = ""):
        self.is_safe = is_safe
        self.reason = reason
        self.nsfw_score = 0.0
        self.gore_score = 0.0

    def __bool__(self) -> bool:
        return self.is_safe


class ContentFilterService:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("enabled", False))
        if self.enabled:
            logger.warning(
                "content_filter.enabled=true but this service is a stub — "
                "all images pass through. Wire up a real detector before going public."
            )

    def check_image(self, image: np.ndarray) -> FilterResult:
        return FilterResult(is_safe=True)
