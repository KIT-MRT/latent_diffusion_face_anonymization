"""Anonymization service wrapping LDFA pipeline."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import Image
import numpy as np
import tempfile
import base64
import io

from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_combined_body_and_lp,
)
from diffusion_face_anonymisation.body import add_body_cutout_and_mask_img

logger = logging.getLogger(__name__)


class AnonymizationService:
    """Service for running anonymization on images."""
    
    def __init__(self, config: dict, detector_manager, endpoint_pool=None):
        self.config = config
        self.detector_manager = detector_manager
        self.endpoint_pool = endpoint_pool
        self.default_methods = {
            'body': config.get('default_body', 'pixel'),
            'lp': config.get('default_lp', 'pixel'),
            'face': config.get('default_face', 'pixel'),
        }
    
    def anonymize(
        self,
        image: Image.Image,
        targets: List[str],
        methods: Dict[str, str],
        detections: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[Image.Image, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Anonymize image with specified targets and methods.
        
        Args:
            image: Input image (RGB)
            targets: List of targets to anonymize
            methods: Dict mapping target -> method
            detections: Pre-computed detections (optional, skips detection)
            
        Returns:
            Tuple of (anonymized_image, detections, metadata)
        """
        start_time = time.time()
        
        # Convert to numpy for processing
        image_np = np.array(image)
        
        # Run detection if not provided
        if detections is None:
            # Save to temp file for detector
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f)
                temp_path = Path(f.name)
            
            try:
                detections = self.detector_manager.detect_all(temp_path, targets)
            finally:
                temp_path.unlink()
        
        # Get anonymization functions
        body_method = methods.get('body', self.default_methods['body'])
        lp_method = methods.get('lp', self.default_methods['lp'])
        
        body_func = define_anon_function(body_method)
        lp_func = define_anon_function(lp_method)
        
        if body_func is None:
            raise ValueError(f"Unknown body method: {body_method}")
        if lp_func is None:
            raise ValueError(f"Unknown LP method: {lp_method}")
        
        # Get detections
        bodies = detections.get('body', [])
        license_plates = detections.get('lp', [])
        
        # Skip if nothing to anonymize
        if not bodies and not license_plates:
            logger.info("No objects to anonymize")
            return image, detections, {
                'processing_time_ms': (time.time() - start_time) * 1000,
                'methods_used': {'body': body_method, 'lp': lp_method},
                'objects_anonymized': {'bodies': 0, 'license_plates': 0},
            }
        
        # Anonymize
        logger.info(f"Anonymizing {len(bodies)} bodies and {len(license_plates)} license plates")
        
        try:
            # Save to temp file (required by anonymize_combined_body_and_lp)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f)
                temp_path = Path(f.name)
            
            try:
                anon_img, bodies, license_plates = anonymize_combined_body_and_lp(
                    image_file=temp_path,
                    bodies=bodies,
                    license_plates=license_plates,
                    body_anon_function=body_func,
                    lp_anon_function=lp_func,
                    endpoint_pool=self.endpoint_pool,
                )
            finally:
                temp_path.unlink()
            
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            raise
        
        processing_time = (time.time() - start_time) * 1000
        
        metadata = {
            'processing_time_ms': processing_time,
            'methods_used': {'body': body_method, 'lp': lp_method},
            'objects_anonymized': {
                'bodies': len(bodies),
                'license_plates': len(license_plates),
            }
        }
        
        return anon_img, detections, metadata
    
    def get_detection_data_for_visualization(
        self,
        detections: Dict[str, List[Any]],
        image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Extract visualization data from detections.
        
        Returns dict with:
        - masks: base64 encoded masks
        - bboxes: bounding boxes
        - confidences: confidence scores
        """
        viz_data = {
            'body': [],
            'lp': [],
            'face': [],
        }
        
        image_np = np.array(image)
        
        # Process bodies
        for body in detections.get('body', []):
            body_data = {
                'bbox': self._get_body_bbox(body),
                'confidence': getattr(body, 'confidence', 1.0),
            }
            
            # Get mask if available
            if hasattr(body, 'body_mask_image') and body.body_mask_image:
                mask_bytes = self._image_to_base64(body.body_mask_image)
                body_data['mask'] = mask_bytes
            
            viz_data['body'].append(body_data)
        
        # Process license plates
        for lp in detections.get('lp', []):
            lp_data = {
                'bbox': self._get_lp_bbox(lp),
                'confidence': getattr(lp, 'confidence', 1.0),
            }
            
            # Create mask from oriented bbox
            mask = self._create_lp_mask(lp, image.size)
            if mask:
                mask_bytes = self._image_to_base64(mask)
                lp_data['mask'] = mask_bytes
            
            viz_data['lp'].append(lp_data)
        
        return viz_data
    
    def _get_body_bbox(self, body) -> List[float]:
        """Extract bounding box from Body object."""
        if hasattr(body, 'bounding_box'):
            return body.bounding_box.get_bbox()
        # Fallback: compute from mask
        if hasattr(body, 'body_mask_image'):
            mask = np.array(body.body_mask_image)
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                return [
                    float(coords[1].min()),
                    float(coords[0].min()),
                    float(coords[1].max()),
                    float(coords[0].max()),
                ]
        return [0, 0, 0, 0]
    
    def _get_lp_bbox(self, lp) -> List[float]:
        """Extract bounding box from LicensePlate object."""
        if hasattr(lp, 'oriented_bbox'):
            bbox = lp.oriented_bbox
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            return [
                float(x_coords.min()),
                float(y_coords.min()),
                float(x_coords.max()),
                float(y_coords.max()),
            ]
        return [0, 0, 0, 0]
    
    def _create_lp_mask(self, lp, image_size: Tuple[int, int]) -> Optional[Image.Image]:
        """Create mask from license plate oriented bbox."""
        if not hasattr(lp, 'oriented_bbox'):
            return None
        
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # Note: PIL uses (W, H)
        bbox = lp.oriented_bbox.astype(np.int32)
        cv2.fillPoly(mask, [bbox], 255)
        return Image.fromarray(mask)
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
