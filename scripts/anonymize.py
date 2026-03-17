#!/usr/bin/env python3
"""Unified Anonymization Script

This script provides a unified interface for anonymizing faces, bodies, and license plates
in images using various detection methods (SAM3, YOLO, YOLOX) and anonymization techniques
(white, gauss, pixel, lda).

Usage Examples:
    # Anonymize bodies and license plates with SAM3 (default) using pixel method
    python anonymize.py --image_dir /data/images --output_dir /data/output --targets body lp --method pixel

    # Anonymize everything with different methods per target
    python anonymize.py --image_dir /data --output_dir /out --targets face body lp \\
        --face_method lda --body_method gauss --lp_method white

    # Use old YOLO detectors instead of SAM3
    python anonymize.py --image_dir /data --output_dir /out --targets body lp \\
        --detector yolo --method pixel

    # Process with all methods
    python anonymize.py --image_dir /data --output_dir /out --targets body --method all

    # Resume interrupted processing
    python anonymize.py --image_dir /data --output_dir /out --targets body lp --method pixel --resume
"""

import os
os.environ['YOLO_VERBOSE'] = 'False'

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
import queue
import threading
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import detection modules
from diffusion_face_anonymisation.body_detection import BodyDetector
from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
from diffusion_face_anonymisation.body import Body
from diffusion_face_anonymisation.license_plate import LicensePlate
from diffusion_face_anonymisation.face import Face
from diffusion_face_anonymisation.anonymization_functions import define_anon_function
from diffusion_face_anonymisation.io_functions import get_faces_from_file
import diffusion_face_anonymisation.utils as dfa_utils

# Import SAM3 detectors (graceful fallback if not available)
try:
    from diffusion_face_anonymisation.sam3_detection import SAM3BodyDetector, SAM3LicensePlateDetector
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("⚠️  SAM3 not available. Use --detector yolo to use YOLO/YOLOX detectors.")

# Set up logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VALID_TARGETS = ['face', 'body', 'lp']
VALID_DETECTORS = ['sam3', 'yolo']
VALID_METHODS = ['white', 'gauss', 'pixel', 'lda', 'all']


def get_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified anonymization script for faces, bodies, and license plates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Core arguments
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for anonymized images")
    parser.add_argument("--targets", type=str, nargs='+', 
                       choices=VALID_TARGETS, default=['body', 'lp'],
                       help="Which objects to anonymize (default: body lp)")
    
    # Detection options
    parser.add_argument("--detector", type=str, choices=VALID_DETECTORS, 
                       default='sam3',
                       help="Global detector choice (default: sam3)")
    parser.add_argument("--body_detector", type=str, choices=VALID_DETECTORS,
                       help="Override body detector (sam3 or yolo)")
    parser.add_argument("--lp_detector", type=str, choices=VALID_DETECTORS,
                       help="Override license plate detector (sam3 or yolox)")
    parser.add_argument("--body_threshold", type=float, default=0.5,
                       help="Body detection confidence threshold (default: 0.5)")
    parser.add_argument("--lp_threshold", type=float, default=0.5,
                       help="License plate detection confidence threshold (default: 0.5)")
    
    # Anonymization options
    parser.add_argument("--method", type=str, choices=VALID_METHODS, 
                       default='white',
                       help="Global anonymization method (default: white)")
    parser.add_argument("--face_method", type=str, 
                       choices=['white', 'gauss', 'pixel', 'lda'],
                       help="Override anonymization method for faces")
    parser.add_argument("--body_method", type=str,
                       choices=['white', 'gauss', 'pixel', 'lda'],
                       help="Override anonymization method for bodies")
    parser.add_argument("--lp_method", type=str,
                       choices=['white', 'gauss', 'pixel'],
                       help="Override anonymization method for license plates (no lda)")
    
    # Face detection specific
    parser.add_argument("--mask_dir", type=str,
                       help="Directory with pre-computed face masks (required if target=face)")
    
    # Performance options
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Detection batch size (default: 8)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel anonymization workers (default: 4)")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Enable streaming pipeline mode (default: enabled)")
    parser.add_argument("--no_streaming", action="store_false", dest="streaming",
                       help="Disable streaming mode, use batch processing")
    
    # Resume & filtering
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Skip already processed images")
    parser.add_argument("--check_only", action="store_true", default=False,
                       help="Only show progress statistics, don't process")
    parser.add_argument("--ext", type=str, default="png",
                       help="Image file extension (default: png)")
    
    return parser


def validate_args(args) -> None:
    """Validate argument combinations."""
    # Check SAM3 availability
    if args.detector == 'sam3' and not SAM3_AVAILABLE:
        raise ValueError("SAM3 not available. Install required packages or use --detector yolo")
    
    # Check face + mask_dir
    if 'face' in args.targets and not args.mask_dir:
        raise ValueError("--mask_dir required when target includes 'face'")
    
    # Warn about lda for license plates
    lp_method = args.lp_method or (args.method if args.method != 'all' else None)
    if lp_method == 'lda':
        logger.warning("LDA not supported for license plates, will use 'white' instead")


def resolve_methods(args, target: str) -> List[str]:
    """Resolve which anonymization methods to use for a given target."""
    # Check target-specific override
    target_method = getattr(args, f'{target}_method', None)
    if target_method:
        return [target_method]
    
    # Use global method
    if args.method == 'all':
        if target == 'lp':
            return ['white', 'gauss', 'pixel']  # no lda for LP
        else:
            return ['white', 'gauss', 'pixel', 'lda']
    else:
        # Filter out lda for LP
        if target == 'lp' and args.method == 'lda':
            logger.warning("LDA not supported for license plates, using 'white' instead")
            return ['white']
        return [args.method]


def get_detector_type(args, target: str) -> str:
    """Get the detector type for a specific target."""
    if target == 'face':
        return 'retinaface'  # Always use RetinaFace for faces
    elif target == 'body':
        return args.body_detector or args.detector
    elif target == 'lp':
        return args.lp_detector or args.detector
    return args.detector


class DetectorFactory:
    """Factory for creating detectors based on target and type."""
    
    @staticmethod
    def create_detector(target: str, detector_type: str, **kwargs):
        """Create appropriate detector for target."""
        if target == 'body':
            if detector_type == 'sam3':
                threshold = kwargs.get('threshold', 0.5)
                return SAM3BodyDetector(threshold=threshold)
            else:  # yolo
                batch_size = kwargs.get('batch_size', 8)
                return BodyDetector(batch_size=batch_size)
        
        elif target == 'lp':
            if detector_type == 'sam3':
                threshold = kwargs.get('threshold', 0.5)
                return SAM3LicensePlateDetector(threshold=threshold)
            else:  # yolox
                threshold = kwargs.get('threshold', 0.25)
                return LicensePlateDetector(conf_threshold=threshold)
        
        elif target == 'face':
            # Face detection uses pre-computed masks
            return None
        
        raise ValueError(f"Unknown target: {target}")


def get_image_files(image_dir: Path, ext: str = "png") -> List[Path]:
    """Get all image files from directory."""
    return sorted(image_dir.glob(f"**/*.{ext}"))


def get_targets_string(targets: List[str]) -> str:
    """Convert targets list to string for filename."""
    return '+'.join(sorted(targets))


def get_output_filename(image_stem: str, targets: List[str], method: str) -> str:
    """Generate output filename based on targets and method."""
    targets_str = get_targets_string(targets)
    return f"{image_stem}_anon_{targets_str}_{method}.png"


def check_progress(output_dir: Path, targets: List[str], methods: List[str], 
                  total_images: int) -> Tuple[int, set]:
    """Check processing progress and return completed count and set."""
    print(f"\n{'='*70}")
    print(f"  📊 PROGRESS CHECK")
    print(f"{'='*70}")
    
    if not output_dir.exists():
        print(f"  Output directory does not exist yet.")
        return 0, set()
    
    targets_str = get_targets_string(targets)
    
    # Count per method
    method_counts = {}
    for method in methods:
        pattern = f"*_anon_{targets_str}_{method}.png"
        count = len(list(output_dir.glob(pattern)))
        method_counts[method] = count
    
    # Get completed images (all methods present)
    completed = set()
    for method in methods:
        pattern = f"*_anon_{targets_str}_{method}.png"
        for f in output_dir.glob(pattern):
            # Extract original stem
            stem = f.stem.replace(f"_anon_{targets_str}_{method}", "")
            
            # Check if all methods exist
            has_all = True
            for m in methods:
                expected = output_dir / get_output_filename(stem, targets, m)
                if not expected.exists():
                    has_all = False
                    break
            
            if has_all:
                completed.add(stem)
    
    print(f"  Input images:        {total_images}")
    print(f"{'─'*70}")
    print(f"  📈 By method:")
    for method, count in method_counts.items():
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"     {method:8s}: {count:5d} / {total_images} ({pct:5.1f}%)")
    
    print(f"{'─'*70}")
    print(f"  ✅ Fully processed: {len(completed):5d} / {total_images}")
    remaining = total_images - len(completed)
    print(f"  ⏳ Remaining:        {remaining}")
    print(f"{'='*70}")
    
    return len(completed), completed


def anonymize_image(image: np.ndarray, 
                    detections: Dict[str, List],
                    methods: Dict[str, str],
                    mask_image: Optional[Image.Image] = None) -> np.ndarray:
    """Apply anonymization to image for all targets.
    
    Args:
        image: Input image as numpy array (RGB)
        detections: Dict mapping target -> list of detected objects
        methods: Dict mapping target -> anonymization method
        mask_image: Optional mask image for face detection
        
    Returns:
        Anonymized image as numpy array (RGB)
    """
    from skimage.filters import gaussian
    
    result = image.copy()
    
    # Simple anonymization functions (for non-LDA methods)
    def anonymize_white_roi(roi):
        return np.ones_like(roi) * 255
    
    def anonymize_gauss_roi(roi):
        if len(roi.shape) == 3:
            return gaussian(roi, preserve_range=True, sigma=3, channel_axis=-1).astype(np.uint8)
        return gaussian(roi, preserve_range=True, sigma=3).astype(np.uint8)
    
    def anonymize_pixelize_roi(roi, pixels_per_block=8):
        h, w = roi.shape[:2]
        if h < pixels_per_block or w < pixels_per_block:
            return roi
        small_h = h // pixels_per_block
        small_w = w // pixels_per_block
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Process in order: face -> body -> lp (lp on top to avoid being covered by body)
    target_order = ['face', 'body', 'lp']
    
    for target in target_order:
        if target not in detections:
            continue
        
        objects = detections[target]
        method = methods.get(target, 'white')
        
        if target == 'face':
            # Face anonymization - use existing pipeline
            anon_func = define_anon_function(method)
            for face in objects:
                if method == 'lda':
                    face = anon_func(obj=face, img=mask_image)
                else:
                    face = anon_func(obj=face)
                result = face.add_anon_face_to_image(result)
        
        elif target == 'body':
            # Body anonymization - direct ROI approach for non-LDA
            if method == 'lda':
                # Use existing pipeline for LDA
                anon_func = define_anon_function(method)
                for body in objects:
                    pil_image = Image.fromarray(result)
                    body = anon_func(obj=body, img=pil_image)
                    result = body.add_anon_body_to_image(result)
            else:
                # Direct ROI approach for white/gauss/pixel
                anon_roi_func = {
                    'white': anonymize_white_roi,
                    'gauss': anonymize_gauss_roi,
                    'pixel': anonymize_pixelize_roi
                }.get(method, anonymize_white_roi)
                
                for body in objects:
                    mask = body.body_mask
                    if mask is None:
                        continue
                    
                    # Normalize mask - handle both grayscale and RGB masks
                    if isinstance(mask, np.ndarray):
                        # Convert RGB mask to grayscale if needed (from YOLO)
                        if len(mask.shape) == 3:
                            # RGB mask - convert to grayscale by taking first channel
                            mask = mask[:, :, 0]
                        
                        if mask.max() > 1:
                            mask = (mask > 127).astype(np.uint8)
                        else:
                            mask = (mask > 0.5).astype(np.uint8)
                    
                    # Find bounding box of mask
                    y_indices, x_indices = np.where(mask > 0)
                    if len(y_indices) == 0 or len(x_indices) == 0:
                        continue
                    
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(result.shape[1], x_max)
                    y_max = min(result.shape[0], y_max)
                    
                    if x_max > x_min and y_max > y_min:
                        roi = result[y_min:y_max, x_min:x_max]
                        anon_roi = anon_roi_func(roi)
                        result[y_min:y_max, x_min:x_max] = anon_roi
        
        elif target == 'lp':
            # License plate anonymization - direct ROI approach
            anon_roi_func = {
                'white': anonymize_white_roi,
                'gauss': anonymize_gauss_roi,
                'pixel': anonymize_pixelize_roi
            }.get(method, anonymize_white_roi)
            
            for lp in objects:
                aabb = lp.aabb
                y_min, y_max, x_min, x_max = aabb
                
                x1 = max(0, x_min)
                y1 = max(0, y_min)
                x2 = min(result.shape[1], x_max)
                y2 = min(result.shape[0], y_max)
                
                if x2 > x1 and y2 > y1:
                    roi = result[y1:y2, x1:x2]
                    anon_roi = anon_roi_func(roi)
                    result[y1:y2, x1:x2] = anon_roi
    
    return result


def process_single_image(image_path: Path, 
                         output_dir: Path,
                         targets: List[str],
                         methods_per_target: Dict[str, List[str]],
                         detectors: Dict[str, Any],
                         mask_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Process a single image with detection and anonymization.
    
    Returns:
        Statistics dict with detection counts
    """
    stem = image_path.stem
    
    # Read image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.warning(f"Could not read {image_path}")
        return {}
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Detect all targets
    detections = {}
    stats = defaultdict(int)
    
    for target in targets:
        if target == 'face':
            # Load from pre-computed masks
            if mask_dir:
                mask_file = mask_dir / f"{stem}_mask.png"
                if mask_file.exists():
                    faces = get_faces_from_file(mask_file)
                    # Add cutout and mask
                    from diffusion_face_anonymisation.face import add_face_cutout_and_mask_img
                    faces = add_face_cutout_and_mask_img(faces=faces, image=image_rgb)
                    detections['face'] = faces
                    stats['faces_detected'] = len(faces)
                else:
                    detections['face'] = []
                    stats['faces_detected'] = 0
        
        elif target == 'body':
            detector = detectors.get('body')
            if detector:
                # BodyDetector (YOLO) has body_detect_in_image, SAM3BodyDetector has detect
                if hasattr(detector, 'body_detect_in_image'):
                    bodies = detector.body_detect_in_image(image_path)
                else:
                    bodies = detector.detect(image_path)
                # Add cutout and mask (only needed for YOLO, SAM3 already has it)
                if hasattr(detector, 'body_detect_in_image'):
                    from diffusion_face_anonymisation.body import add_body_cutout_and_mask_img
                    bodies = add_body_cutout_and_mask_img(bodies, image_rgb)
                detections['body'] = bodies
                stats['bodies_detected'] = len(bodies)
        
        elif target == 'lp':
            detector = detectors.get('lp')
            if detector:
                lps = detector.detect(image_path)
                detections['lp'] = lps
                stats['lps_detected'] = len(lps)
    
    # Generate all method combinations
    # We need to create one output per unique method combination
    all_methods = set()
    for target in targets:
        all_methods.update(methods_per_target.get(target, []))
    
    # For each unique method, create output
    for method in all_methods:
        output_filename = get_output_filename(stem, targets, method)
        output_path = output_dir / output_filename
        
        # Skip if exists (for resume)
        if output_path.exists():
            continue
        
        # Build method dict (same method for all targets for now)
        # TODO: Support different methods per target in single output
        method_dict = {target: method for target in targets}
        
        # Load mask image for face+lda if needed
        mask_image = None
        if 'face' in targets and method == 'lda' and mask_dir:
            mask_file = mask_dir / f"{stem}_mask.png"
            if mask_file.exists():
                mask_image = Image.open(image_path)
        
        # Anonymize
        try:
            anon_image = anonymize_image(image_rgb, detections, method_dict, mask_image)
            
            # Save
            anon_bgr = cv2.cvtColor(anon_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), anon_bgr)
            
            stats[f'method_{method}_success'] = 1
        except Exception as e:
            import traceback
            logger.error(f"Error anonymizing {image_path} with {method}: {e}")
            logger.debug(traceback.format_exc())
            stats[f'method_{method}_error'] = 1
    
    return stats


def run_streaming_pipeline(image_files: List[Path],
                           output_dir: Path,
                           targets: List[str],
                           methods_per_target: Dict[str, List[str]],
                           detectors: Dict[str, Any],
                           args) -> Dict[str, Any]:
    """Run detection and anonymization in streaming/pipeline mode."""
    print(f"\n{'='*70}")
    print(f"  🚀 STREAMING MODE")
    print(f"{'='*70}")
    
    # Collect all unique methods across all targets
    all_methods = set()
    for methods in methods_per_target.values():
        all_methods.update(methods)
    all_methods = sorted(all_methods)
    
    total_outputs = len(image_files) * len(all_methods)
    stats = {
        'total_images': len(image_files),
        'total_outputs': total_outputs,
        'method_counts': {m: 0 for m in all_methods},
        'detection_stats': defaultdict(int),
        'start_time': time.time()
    }
    
    # Simple sequential processing (no threading for now to avoid complexity)
    pbar = tqdm(total=total_outputs, desc="Processing", unit="img")
    
    for image_path in image_files:
        img_stats = process_single_image(
            image_path, output_dir, targets, methods_per_target, 
            detectors, Path(args.mask_dir) if args.mask_dir else None
        )
        
        # Update stats
        for key, val in img_stats.items():
            if key.startswith('method_'):
                method = key.split('_')[1]
                if 'success' in key:
                    stats['method_counts'][method] += 1
                    pbar.update(1)
            elif key in ['faces_detected', 'bodies_detected', 'lps_detected']:
                stats['detection_stats'][key] += val
                if val > 0:
                    stats['detection_stats'][f"images_with_{key.replace('_detected', '')}"] += 1
    
    pbar.close()
    
    stats['end_time'] = time.time()
    stats['total_time_seconds'] = stats['end_time'] - stats['start_time']
    stats['throughput_images_per_sec'] = len(image_files) / stats['total_time_seconds']
    stats['avg_time_per_image_ms'] = (stats['total_time_seconds'] / len(image_files)) * 1000
    
    return stats


def save_statistics(stats: Dict[str, Any], output_dir: Path, args) -> None:
    """Save processing statistics to JSON file."""
    stats_file = output_dir / "stats.json"
    
    # Build detector info
    detector_info = {}
    for target in args.targets:
        detector_type = get_detector_type(args, target)
        detector_info[target] = detector_type
    
    # Build methods info
    methods_info = {}
    for target in args.targets:
        methods = resolve_methods(args, target)
        methods_info[target] = methods
    
    output_data = {
        "input_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "total_images": stats.get('total_images', 0),
        "targets": args.targets,
        "detectors": detector_info,
        "methods": methods_info,
        "processing_stats": {
            "total_outputs": stats.get('total_outputs', 0),
            "total_time_seconds": round(stats.get('total_time_seconds', 0), 2),
            "throughput_images_per_sec": round(stats.get('throughput_images_per_sec', 0), 2),
            "avg_time_per_image_ms": round(stats.get('avg_time_per_image_ms', 0), 2),
        },
        "method_counts": stats.get('method_counts', {}),
        "detection_stats": dict(stats.get('detection_stats', {})),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(stats_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n📊 Statistics saved to: {stats_file}")


def main():
    """Main entry point."""
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    # Setup paths
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"  🎯 UNIFIED ANONYMIZATION")
    print(f"{'='*70}")
    print(f"  📁 Input:     {args.image_dir}")
    print(f"  📁 Output:    {args.output_dir}")
    print(f"  🎯 Targets:   {', '.join(args.targets)}")
    print(f"  🔍 Detector:  {args.detector}")
    print(f"  ⚙️  Method:    {args.method}")
    if args.streaming:
        print(f"  📊 Mode:      Streaming")
    else:
        print(f"  📊 Mode:      Batch")
    print(f"{'='*70}")
    
    # Get image files
    images = get_image_files(image_dir, args.ext)
    if not images:
        print(f"❌ No images found in {image_dir} with extension .{args.ext}")
        return 1
    
    print(f"  Found {len(images)} images")
    
    # Resolve methods per target
    methods_per_target = {}
    for target in args.targets:
        methods_per_target[target] = resolve_methods(args, target)
    
    # Collect all unique methods for progress tracking
    all_methods = set()
    for methods in methods_per_target.values():
        all_methods.update(methods)
    all_methods = sorted(all_methods)
    
    print(f"  Methods: {', '.join(all_methods)}")
    print(f"{'='*70}")
    
    # Check progress
    if args.check_only or args.resume:
        completed_count, completed_set = check_progress(
            output_dir, args.targets, all_methods, len(images)
        )
        
        if args.check_only:
            print("\n✅ --check_only specified, exiting.")
            return 0
        
        if args.resume and completed_set:
            original_count = len(images)
            images = [img for img in images if img.stem not in completed_set]
            skipped = original_count - len(images)
            print(f"\n🔄 --resume enabled: skipping {skipped} already-processed images")
            print(f"   Processing {len(images)} remaining images\n")
    
    if not images:
        print("✅ All images already processed!")
        return 0
    
    # Initialize detectors
    print(f"\n📦 Loading detectors...")
    detectors = {}
    
    for target in args.targets:
        if target == 'face':
            continue  # Face uses pre-computed masks
        
        detector_type = get_detector_type(args, target)
        print(f"  {target:6s}: {detector_type}")
        
        try:
            if target == 'body':
                detectors['body'] = DetectorFactory.create_detector(
                    'body', detector_type,
                    threshold=args.body_threshold,
                    batch_size=args.batch_size
                )
            elif target == 'lp':
                detectors['lp'] = DetectorFactory.create_detector(
                    'lp', detector_type,
                    threshold=args.lp_threshold,
                    batch_size=args.batch_size
                )
        except Exception as e:
            print(f"❌ Error loading {target} detector ({detector_type}): {e}")
            return 1
    
    print(f"✅ Detectors loaded\n")
    
    # Run processing
    start_time = time.time()
    
    if args.streaming:
        stats = run_streaming_pipeline(
            images, output_dir, args.targets, methods_per_target, detectors, args
        )
    else:
        # Batch mode not implemented yet, fall back to streaming
        print("⚠️  Batch mode not yet implemented, using streaming mode")
        stats = run_streaming_pipeline(
            images, output_dir, args.targets, methods_per_target, detectors, args
        )
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  ✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  ⏱️  Total time:    {elapsed/60:.1f} minutes ({elapsed:.1f}s)")
    print(f"  🚀 Throughput:   {stats.get('throughput_images_per_sec', 0):.2f} images/second")
    print(f"  📈 Per method:")
    for method, count in stats.get('method_counts', {}).items():
        print(f"     {method:8s}: {count} outputs")
    
    if stats.get('detection_stats'):
        print(f"  🔍 Detections:")
        det_stats = stats['detection_stats']
        for key in ['bodies_detected', 'lps_detected', 'faces_detected']:
            if key in det_stats:
                print(f"     {key:20s}: {det_stats[key]}")
    
    print(f"{'='*70}")
    
    # Save statistics
    save_statistics(stats, output_dir, args)
    
    return 0


if __name__ == "__main__":
    exit(main())
