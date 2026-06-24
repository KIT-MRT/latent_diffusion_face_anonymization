"""Detector manager for SAM3 and YOLO detectors."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DetectorManager:
    """Manages detector initialization and batch detection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.detector_type = config.get('default', 'sam3')
        self.sam3_config = config.get('sam3', {})
        self.yolo_config = config.get('yolo', {})
        
        self.body_detector = None
        self.lp_detector = None
        self._initialized = False
    
    def initialize(self, detector_type: Optional[str] = None):
        """Initialize detectors."""
        detector_type = detector_type or self.detector_type
        
        logger.info(f"Initializing {detector_type} detectors...")
        
        if detector_type == 'sam3':
            self._initialize_sam3()
        elif detector_type == 'yolo':
            self._initialize_yolo()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        self._initialized = True
        logger.info("Detectors initialized successfully")
    
    def _initialize_sam3(self):
        """Initialize SAM3 detectors."""
        try:
            from diffusion_face_anonymisation.sam3_detection import (
                SAM3BodyDetector,
                SAM3LicensePlateDetector
            )
            
            body_threshold = self.sam3_config.get('body_threshold', 0.5)
            mask_threshold = self.sam3_config.get('mask_threshold', 0.5)
            lp_threshold = self.sam3_config.get('lp_threshold', 0.5)
            
            logger.info("Loading SAM3 body detector...")
            self.body_detector = SAM3BodyDetector(
                threshold=body_threshold,
                mask_threshold=mask_threshold
            )
            
            logger.info("Loading SAM3 license plate detector...")
            self.lp_detector = SAM3LicensePlateDetector(
                threshold=lp_threshold,
                mask_threshold=mask_threshold
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize SAM3 detectors: {e}")
            raise
    
    def _initialize_yolo(self):
        """Initialize YOLO detectors."""
        try:
            from diffusion_face_anonymisation.body_detection import BodyDetector
            from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
            
            body_batch_size = self.yolo_config.get('body_batch_size', 8)
            lp_conf_threshold = self.yolo_config.get('lp_conf_threshold', 0.25)
            
            logger.info("Loading YOLO body detector...")
            self.body_detector = BodyDetector(batch_size=body_batch_size)
            
            logger.info("Loading YOLOX license plate detector...")
            self.lp_detector = LicensePlateDetector(conf_threshold=lp_conf_threshold)
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detectors: {e}")
            raise
    
    def detect_bodies(self, image_path: Path) -> List[Any]:
        """Detect bodies in image."""
        if not self._initialized or self.body_detector is None:
            raise RuntimeError("Body detector not initialized")
        
        return self.body_detector.detect(image_path)
    
    def detect_license_plates(self, image_path: Path) -> List[Any]:
        """Detect license plates in image."""
        if not self._initialized or self.lp_detector is None:
            raise RuntimeError("License plate detector not initialized")
        
        return self.lp_detector.detect(image_path)
    
    def detect_all(self, image_path: Path, targets: List[str]) -> Dict[str, List[Any]]:
        """
        Detect all requested targets in image.
        
        Args:
            image_path: Path to image
            targets: List of targets ['body', 'lp', 'face']
            
        Returns:
            Dict mapping target -> list of detections
        """
        results = {}
        
        if 'body' in targets and self.body_detector:
            logger.info(f"Detecting bodies in {image_path}")
            results['body'] = self.body_detector.detect(image_path)
            logger.info(f"Found {len(results['body'])} bodies")
        
        if 'lp' in targets and self.lp_detector:
            logger.info(f"Detecting license plates in {image_path}")
            results['lp'] = self.lp_detector.detect(image_path)
            logger.info(f"Found {len(results['lp'])} license plates")
        
        # Face detection not implemented in detector manager yet
        # Would use RetinaFace from existing code
        if 'face' in targets:
            logger.warning("Face detection not yet implemented in DetectorManager")
            results['face'] = []
        
        return results
    
    def is_initialized(self) -> bool:
        """Check if detectors are initialized."""
        return self._initialized
