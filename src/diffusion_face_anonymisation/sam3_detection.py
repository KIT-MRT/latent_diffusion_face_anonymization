"""SAM3-based detectors for body and license plate detection.

This module provides alternative detectors using the SAM3 (Segment Anything 3)
model from Meta, which uses text prompts for open-vocabulary segmentation.

Usage:
    from diffusion_face_anonymisation.sam3_detection import SAM3BodyDetector, SAM3LicensePlateDetector
    
    body_detector = SAM3BodyDetector(threshold=0.5)
    bodies = body_detector.detect("image.jpg")
    
    lp_detector = SAM3LicensePlateDetector(threshold=0.5)
    license_plates = lp_detector.detect("image.jpg")
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch

from diffusion_face_anonymisation.body import Body
from diffusion_face_anonymisation.license_plate import LicensePlate

logger = logging.getLogger(__name__)


class SAM3BodyDetector:
    """Body detector using SAM3 with text prompt 'person'."""
    
    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5, device: str = None):
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading SAM3 model for body detection on {self.device}...")
        from transformers import Sam3Model, Sam3Processor
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model.eval()
        logger.info("SAM3 body detector loaded successfully.")
    
    def detect(self, image_path: Path) -> List[Body]:
        """Detect persons in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of Body objects with segmentation masks
        """
        logger.info(f"Detecting bodies in {image_path} with SAM3...")
        
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        inputs = self.processor(images=image, text="person", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(h, w)]
        )[0]
        
        bodies = []
        if "masks" in results and len(results["masks"]) > 0:
            masks = results["masks"]
            boxes = results.get("boxes", [])
            scores = results.get("scores", torch.ones(len(masks)))
            
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                body = Body(mask_np)
                body.body_mask_image = Image.fromarray(mask_np)
                body.body_cutout = self._extract_from_mask(image_array, mask_np)
                bodies.append(body)
                
                if i < len(boxes):
                    logger.info(f"  Detected person {i+1}: box={boxes[i].tolist()}, score={scores[i]:.3f}")
                else:
                    logger.info(f"  Detected person {i+1}: score={scores[i]:.3f}")
        
        logger.info(f"SAM3 detected {len(bodies)} persons in {image_path}")
        return bodies
    
    def detect_batch(self, image_paths: List[Path]) -> Dict[Path, List[Body]]:
        """Detect bodies in multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to lists of Body objects
        """
        results = {}
        for img_path in image_paths:
            results[img_path] = self.detect(img_path)
        return results
    
    def _extract_from_mask(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Extract region from image using mask."""
        cutout = np.zeros_like(image)
        idx = mask > 0
        cutout[idx] = image[idx]
        return Image.fromarray(cutout)


class SAM3LicensePlateDetector:
    """License plate detector using SAM3 with text prompt 'license plate'."""
    
    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5, device: str = None):
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading SAM3 model for license plate detection on {self.device}...")
        from transformers import Sam3Model, Sam3Processor
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model.eval()
        logger.info("SAM3 license plate detector loaded successfully.")
    
    def detect(self, image_path: Path) -> List[LicensePlate]:
        """Detect license plates in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of LicensePlate objects with oriented bounding boxes
        """
        logger.info(f"Detecting license plates in {image_path} with SAM3...")
        
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        inputs = self.processor(images=image, text="license plate", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(h, w)]
        )[0]
        
        license_plates = []
        if "masks" in results and len(results["masks"]) > 0:
            masks = results["masks"]
            boxes = results.get("boxes", [])
            scores = results.get("scores", torch.ones(len(masks)))
            
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                
                if i < len(boxes):
                    box = boxes[i].tolist()
                    x1, y1, x2, y2 = box
                    oriented_bbox = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
                    lp = LicensePlate(oriented_bbox)
                    lp.set_lp_cutout(image_array)
                    license_plates.append(lp)
                    logger.info(f"  Detected license plate {i+1}: box={box}, score={scores[i]:.3f}")
                else:
                    oriented_bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]])
                    lp = LicensePlate(oriented_bbox)
                    lp.set_lp_cutout(image_array)
                    license_plates.append(lp)
                    logger.info(f"  Detected license plate {i+1}: score={scores[i]:.3f}")
        
        logger.info(f"SAM3 detected {len(license_plates)} license plates in {image_path}")
        return license_plates
    
    def detect_batch(self, image_paths: List[Path]) -> Dict[Path, List[LicensePlate]]:
        """Detect license plates in multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to lists of LicensePlate objects
        """
        results = {}
        for img_path in image_paths:
            results[img_path] = self.detect(img_path)
        return results


def detect_with_sam3(image_path: Path, threshold: float = 0.5) -> Dict[str, Any]:
    """Convenience function to run both body and license plate detection.
    
    Args:
        image_path: Path to the image file
        threshold: Detection threshold for both detectors
        
    Returns:
        Dictionary with 'bodies' and 'license_plates' lists
    """
    body_detector = SAM3BodyDetector(threshold=threshold)
    lp_detector = SAM3LicensePlateDetector(threshold=threshold)
    
    bodies = body_detector.detect(image_path)
    license_plates = lp_detector.detect(image_path)
    
    return {
        "bodies": bodies,
        "license_plates": license_plates
    }
