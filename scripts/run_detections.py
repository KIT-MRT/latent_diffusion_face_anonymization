#!/usr/bin/env python3
"""Run both existing and SAM3 detectors on test images.

This script runs:
- Existing detectors: BodyDetector (YOLO12L), LicensePlateDetector (YOLOX)
- SAM3 detectors: SAM3BodyDetector, SAM3LicensePlateDetector

Results are saved to data/detections/{method}/
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

DATA_DIR = Path("data")
TEST_IMAGES_DIR = DATA_DIR / "test_images"
DETECTIONS_DIR = DATA_DIR / "detections"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_image_files():
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = []
    for f in TEST_IMAGES_DIR.iterdir():
        if f.suffix.lower() in image_extensions:
            files.append(f)
    return sorted(files)


def bbox_from_aabb(aabb):
    """Convert (y_min, y_max, x_min, x_max) to [x1, y1, x2, y2]."""
    y_min, y_max, x_min, x_max = aabb
    return [x_min, y_min, x_max, y_max]


def run_existing_detectors(image_files):
    """Run existing YOLO and YOLOX detectors."""
    from diffusion_face_anonymisation.body_detection import BodyDetector
    from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
    
    logger.info("Initializing existing detectors...")
    body_detector = BodyDetector(batch_size=1)
    lp_detector = LicensePlateDetector()
    
    results = {}
    
    for img_path in tqdm(image_files, desc="Running existing detectors"):
        img_results = {"humans": [], "license_plates": []}
        
        try:
            bodies = body_detector.body_detect_in_image(img_path)
            for body in bodies:
                aabb = body.body_mask.shape
                if hasattr(body, 'body_mask_image') and body.body_mask_image is not None:
                    mask = np.array(body.body_mask_image)
                    h, w = mask.shape
                    if h > 0 and w > 0:
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            img_results["humans"].append({
                                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                                "height_px": float(y_max - y_min),
                                "height_pct": float((y_max - y_min) / mask.shape[0]) if mask.shape[0] > 0 else 0
                            })
        except Exception as e:
            logger.error(f"Error detecting bodies in {img_path.name}: {e}")
        
        try:
            license_plates = lp_detector.detect(img_path)
            for lp in license_plates:
                bbox = bbox_from_aabb(lp.aabb)
                img_results["license_plates"].append({
                    "bbox": bbox,
                    "width_px": float(bbox[2] - bbox[0]),
                    "height_px": float(bbox[3] - bbox[1]),
                    "area_pct": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1000000
                })
        except Exception as e:
            logger.error(f"Error detecting license plates in {img_path.name}: {e}")
        
        results[img_path.name] = img_results
    
    return results


def run_sam3_detectors(image_files):
    """Run SAM3-based detectors."""
    from diffusion_face_anonymisation.sam3_detection import SAM3BodyDetector, SAM3LicensePlateDetector
    
    logger.info("Initializing SAM3 detectors...")
    body_detector = SAM3BodyDetector(threshold=0.5)
    lp_detector = SAM3LicensePlateDetector(threshold=0.5)
    
    results = {}
    
    for img_path in tqdm(image_files, desc="Running SAM3 detectors"):
        img_results = {"humans": [], "license_plates": []}
        
        try:
            bodies = body_detector.detect(img_path)
            for body in bodies:
                mask = body.body_mask
                if mask is not None:
                    h, w = mask.shape
                    if h > 0 and w > 0:
                        y_indices, x_indices = np.where(mask > 0)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            img_results["humans"].append({
                                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                                "height_px": float(y_max - y_min),
                                "height_pct": float((y_max - y_min) / h) if h > 0 else 0
                            })
        except Exception as e:
            logger.error(f"Error detecting bodies with SAM3 in {img_path.name}: {e}")
        
        try:
            license_plates = lp_detector.detect(img_path)
            for lp in license_plates:
                bbox = bbox_from_aabb(lp.aabb)
                img_results["license_plates"].append({
                    "bbox": bbox,
                    "width_px": float(bbox[2] - bbox[0]),
                    "height_px": float(bbox[3] - bbox[1]),
                    "area_pct": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1000000
                })
        except Exception as e:
            logger.error(f"Error detecting license plates with SAM3 in {img_path.name}: {e}")
        
        results[img_path.name] = img_results
    
    return results


def save_results(results, method: str):
    """Save detection results to JSON file."""
    output_dir = DETECTIONS_DIR / method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "detections.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved {method} detections to {output_file}")
    return output_file


def main():
    print("=" * 60)
    print("RUNNING DETECTORS ON TEST IMAGES")
    print("=" * 60)
    
    image_files = get_image_files()
    print(f"\nFound {len(image_files)} images to process")
    
    if not image_files:
        print("No images found!")
        return
    
    print("\n--- Running existing detectors (YOLO + YOLOX) ---")
    existing_results = run_existing_detectors(image_files)
    save_results(existing_results, "existing")
    
    print("\n--- Running SAM3 detectors ---")
    sam3_results = run_sam3_detectors(image_files)
    save_results(sam3_results, "sam3")
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE!")
    print("=" * 60)
    
    print("\n--- Summary ---")
    total_humans_existing = sum(len(r["humans"]) for r in existing_results.values())
    total_lp_existing = sum(len(r["license_plates"]) for r in existing_results.values())
    total_humans_sam3 = sum(len(r["humans"]) for r in sam3_results.values())
    total_lp_sam3 = sum(len(r["license_plates"]) for r in sam3_results.values())
    
    print(f"Existing detectors: {total_humans_existing} humans, {total_lp_existing} license plates")
    print(f"SAM3 detectors: {total_humans_sam3} humans, {total_lp_sam3} license plates")


if __name__ == "__main__":
    main()
