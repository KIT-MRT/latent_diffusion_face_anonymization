"""Unified streaming pipeline for anonymization.

This module provides a producer/consumer architecture that:
- Producer: batch detects bodies/LPs, puts results in queue
- Consumer: takes from queue, anonymizes, saves

This enables parallelism: while one batch is being anonymized, 
the next batch is being detected.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_combined_body_and_lp,
)
from diffusion_face_anonymisation.body import add_body_cutout_and_mask_img

logger = logging.getLogger(__name__)


def create_detector(target: str, detector_type: str, **kwargs):
    """Factory to create any detector.
    
    Args:
        target: 'body' or 'lp'
        detector_type: 'sam3' or 'yolo' (yolo/yolox for body/LP respectively)
        **kwargs: Additional args passed to detector constructor
                 - body + sam3: threshold, mask_threshold
                 - body + yolo: batch_size
                 - lp + sam3: threshold, mask_threshold
                 - lp + yolox: conf_threshold
    
    Returns:
        Detector instance with .detect() method
    """
    # Extract common args - SAM3 doesn't use these
    threshold = kwargs.pop('threshold', None)
    batch_size = kwargs.pop('batch_size', 8)
    
    if target == 'body':
        if detector_type == 'sam3':
            from diffusion_face_anonymisation.sam3_detection import SAM3BodyDetector
            if threshold:
                return SAM3BodyDetector(threshold=threshold)
            return SAM3BodyDetector()
        else:  # yolo
            from diffusion_face_anonymisation.body_detection import BodyDetector
            return BodyDetector(batch_size=batch_size)
    
    elif target == 'lp':
        if detector_type == 'sam3':
            from diffusion_face_anonymisation.sam3_detection import SAM3LicensePlateDetector
            if threshold:
                return SAM3LicensePlateDetector(threshold=threshold)
            return SAM3LicensePlateDetector()
        else:  # yolox
            from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
            conf = threshold if threshold else 0.25
            return LicensePlateDetector(conf_threshold=conf)
    
    raise ValueError(f"Unknown target: {target}")


def detect_batch(detector, image_files: List[Path]) -> Dict[Path, List]:
    """Run batch detection on a list of images.
    
    Args:
        detector: Detector instance with .detect() or .batch_detect() method
        image_files: List of image paths
        
    Returns:
        Dict mapping image_path -> list of detected objects
    """
    # Check if detector supports batch detection
    if hasattr(detector, 'batch_detect'):
        return detector.batch_detect(image_files)
    
    # Fall back to sequential detection
    results = {}
    for img_path in image_files:
        results[img_path] = detector.detect(img_path)
    return results


def process_image_anonymization(
    image_path: Path,
    bodies: List[Any],
    license_plates: List[Any],
    targets: List[str],
    methods: Dict[str, str],
    output_dir: Path,
    endpoint_pool=None,
) -> bool:
    """Anonymize a single image with given detections and methods.
    
    Args:
        image_path: Path to input image
        bodies: List of Body objects
        license_plates: List of LicensePlate objects
        targets: List of targets being processed
        methods: Dict mapping target -> method (e.g., {'body': 'lda', 'lp': 'pixel'})
        output_dir: Output directory
        endpoint_pool: Optional APIEndpointPool for multi-GPU LDA
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"Could not read {image_path}")
            return False
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Get anonymization functions
        body_method = methods.get('body', 'white')
        lp_method = methods.get('lp', 'white')
        
        body_func = define_anon_function(body_method)
        lp_func = define_anon_function(lp_method)
        
        # Anonymize - pass endpoint_pool for multi-GPU LDA
        from diffusion_face_anonymisation.license_plate import LicensePlate
        anon_img, _, _ = anonymize_combined_body_and_lp(
            image_path, bodies, license_plates, body_func, lp_func,
            endpoint_pool=endpoint_pool
        )
        
        # Save with proper naming: {stem}_anon_{targets}_{methods}.png
        stem = image_path.stem
        
        # Build targets string: "body+lp" or just "body" or "lp"
        targets_str = "+".join(sorted(targets))
        
        # Build method suffix: if different methods, use "bodyMethod_lpMethod" format
        if body_method == lp_method:
            method_suffix = body_method
        else:
            method_suffix = f"{body_method}_{lp_method}"
        
        output_path = output_dir / f"{stem}_anon_{targets_str}_{method_suffix}{image_path.suffix}"
        
        anon_bgr = cv2.cvtColor(np.array(anon_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), anon_bgr)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False


def run_streaming_pipeline(
    image_files: List[Path],
    targets: List[str],
    detectors: Dict[str, Any],
    methods: Dict[str, str],
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    endpoint_pool=None,
) -> Dict[str, Any]:
    """Run detection and anonymization in streaming/pipeline mode.
    
    Producer: Batch detects objects, puts results in queue
    Consumer: Takes from queue, anonymizes, saves
    
    Args:
        image_files: List of image paths to process
        targets: List of targets ['body', 'lp']
        detectors: Dict mapping target -> detector instance
        methods: Dict mapping target -> method (e.g., {'body': 'lda', 'lp': 'pixel'})
        output_dir: Output directory
        batch_size: Batch size for detection
        num_workers: Number of consumer threads
        endpoint_pool: Optional APIEndpointPool for multi-GPU LDA
        
    Returns:
        Statistics dict
    """
    body_method = methods.get('body', 'white')
    lp_method = methods.get('lp', 'white')
    
    total_outputs = len(image_files)
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    result_queue = queue.Queue(maxsize=50)
    completed_count = [0]
    start_time = time.time()
    batches_processed = [0]
    
    pbar = tqdm(total=total_outputs, desc="Processing", unit="img")
    
    def producer():
        """Producer: batch detection"""
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch = image_files[start_idx:end_idx]
            
            # Run detection for each target and store separately
            all_detections = {target: {} for target in targets}
            for target in targets:
                if target not in detectors:
                    continue
                detector = detectors[target]
                target_detections = detect_batch(detector, batch)
                all_detections[target].update(target_detections)
            
            # Put results in queue - combine all detections for each image
            for img_path in batch:
                img_bodies = all_detections.get('body', {}).get(img_path, [])
                img_lps = all_detections.get('lp', {}).get(img_path, [])
                result_queue.put((img_path, img_bodies, img_lps))
            
            batches_processed[0] = batch_idx + 1
        
        # Signal completion
        for _ in range(num_workers):
            result_queue.put(None)
    
    def consumer(worker_id: int):
        """Consumer: anonymization worker"""
        while True:
            try:
                item = result_queue.get(timeout=2)
            except queue.Empty:
                continue
            
            if item is None:
                break
            
            img_path, bodies, lps = item
            
            success = process_image_anonymization(
                img_path, bodies, lps,
                targets,
                methods, output_dir,
                endpoint_pool=endpoint_pool
            )
            
            completed_count[0] += 1
            pbar.update(1)
            
            # Update progress
            elapsed = time.time() - start_time
            speed = completed_count[0] / elapsed if elapsed > 0 else 0
            queue_size = result_queue.qsize()
            
            pbar.set_postfix({
                'det': f'{batches_processed[0]}/{total_batches}',
                'queue': queue_size,
                'speed': f'{speed:.1f}/s',
            })
    
    # Start producer
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    
    # Start consumers
    consumer_threads = [
        threading.Thread(target=consumer, args=(i,)) 
        for i in range(num_workers)
    ]
    for t in consumer_threads:
        t.start()
    
    # Wait
    producer_thread.join()
    for t in consumer_threads:
        t.join()
    
    pbar.close()
    
    elapsed = time.time() - start_time
    
    return {
        'total_images': len(image_files),
        'total_outputs': completed_count[0],
        'total_time_seconds': elapsed,
        'throughput_images_per_sec': completed_count[0] / elapsed if elapsed > 0 else 0,
    }


def check_progress(
    output_dir: Path,
    targets: List[str],
    body_method: str,
    lp_method: str,
    total_images: int,
) -> tuple[int, List[Path]]:
    """Check which images have been completed.
    
    Args:
        output_dir: Output directory
        targets: List of targets
        body_method: Body anonymization method
        lp_method: LP anonymization method  
        total_images: Total number of input images
        
    Returns:
        Tuple of (completed_count, list of remaining image paths)
    """
    if not output_dir.exists():
        return 0, []
    
    method_suffix = f"{body_method}_{lp_method}" if body_method != lp_method else body_method
    
    # Find completed images
    completed = set()
    for f in output_dir.glob(f"*_anon_{method_suffix}.png"):
        stem = f.stem.replace(f"_anon_{method_suffix}", "")
        completed.add(stem)
    
    completed_count = len(completed)
    print(f"\nProgress: {completed_count}/{total_images} images completed")
    
    return completed_count, []


def get_image_files(image_dir: Path, ext: str = "png") -> List[Path]:
    """Get all image files from directory."""
    return sorted(image_dir.glob(f"**/*.{ext}"))
