import os
os.environ['YOLO_VERBOSE'] = 'False'

import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian

from diffusion_face_anonymisation.sam3_detection import SAM3LicensePlateDetector
from diffusion_face_anonymisation.license_plate import LicensePlate


def anonymize_white(img):
    return np.ones_like(img) * 255


def anonymize_gauss(img):
    if len(img.shape) == 3:
        return gaussian(img, preserve_range=True, sigma=3, channel_axis=-1).astype(np.uint8)
    return gaussian(img, preserve_range=True, sigma=3).astype(np.uint8)


def anonymize_pixelize(img, pixels_per_block=8):
    h, w = img.shape[:2]
    if h < pixels_per_block or w < pixels_per_block:
        return img
    small_h = h // pixels_per_block
    small_w = w // pixels_per_block
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


ANON_FUNCTIONS = {
    "white": anonymize_white,
    "gauss": anonymize_gauss,
    "pixel": anonymize_pixelize,
}

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_image_files(image_dir: Path, ext: str = "png") -> list[Path]:
    return sorted(image_dir.glob(f"**/*.{ext}"))


def get_completed_images(output_dir: Path, anon_methods: list[str]) -> set:
    if not output_dir.exists():
        return set()
    
    completed = set()
    for f in output_dir.glob("*_anon_white.png"):
        stem = f.stem.replace("_anon_white", "")
        has_all = True
        for method in anon_methods:
            method_file = output_dir / f"{stem}_anon_{method}.png"
            if not method_file.exists():
                has_all = False
                break
        if has_all:
            completed.add(stem)
    return completed


def check_progress(output_dir: Path, anon_methods: list[str], total_input_images: int):
    print(f"\n{'='*70}")
    print(f"  📊 PROGRESS CHECK")
    print(f"{'='*70}")
    
    if not output_dir.exists():
        print(f"  Output directory does not exist yet.")
        return 0, set()
    
    method_counts = {}
    for method in anon_methods:
        count = len(list(output_dir.glob(f"*_anon_{method}.png")))
        method_counts[method] = count
    
    completed_images = get_completed_images(output_dir, anon_methods)
    
    print(f"  Input images:        {total_input_images}")
    print(f"{'─'*70}")
    for method in anon_methods:
        print(f"  {method:12} done:   {method_counts.get(method, 0):>6} ({100*method_counts.get(method, 0)/max(1,total_input_images):>5.1f}%)")
    print(f"{'─'*70}")
    print(f"  Full completions:   {len(completed_images):>6} ({100*len(completed_images)/max(1,total_input_images):>5.1f}%)")
    print(f"{'='*70}\n")
    
    return len(completed_images), completed_images


def anonymize_lp_image(image: np.ndarray, license_plates: list[LicensePlate], anon_function) -> np.ndarray:
    image = image.copy()
    
    for lp in license_plates:
        aabb = lp.aabb
        y_min, y_max, x_min, x_max = aabb
        
        x1 = max(0, x_min)
        y1 = max(0, y_min)
        x2 = min(image.shape[1], x_max)
        y2 = min(image.shape[0], y_max)
        
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            anon_roi = anon_function(roi)
            image[y1:y2, x1:x2] = anon_roi
    
    return image


def process_image(args):
    image_path, output_dir, anon_methods, lp_detector = args
    
    stem = image_path.stem
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not read {image_path}")
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    license_plates = lp_detector.detect(image_path)
    
    if not license_plates:
        logger.info(f"No license plates detected in {image_path.name}")
    
    for method in anon_methods:
        output_path = output_dir / f"{stem}_anon_{method}.png"
        if output_path.exists():
            continue
        
        anon_function = ANON_FUNCTIONS[method]
        anon_image = anonymize_lp_image(image, license_plates, anon_function)
        
        anon_image_bgr = cv2.cvtColor(anon_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), anon_image_bgr)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Anonymize license plates using SAM3")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--anon_function", type=str, default="white", 
                       choices=["white", "gauss", "pixel"])
    parser.add_argument("--ext", type=str, default="png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = get_image_files(image_dir, args.ext)
    print(f"Found {len(images)} images in {image_dir}")
    
    anon_methods = args.anon_function.split(",") if "," in args.anon_function else [args.anon_function]
    
    completed, completed_set = check_progress(output_dir, anon_methods, len(images))
    
    if args.skip_existing and completed > 0:
        images = [img for img in images if img.stem not in completed_set]
        print(f"Skipping {completed} already processed images, {len(images)} remaining")

    if not images:
        print("No images to process!")
        return

    print(f"Processing {len(images)} images with SAM3 license plate detector")
    print(f"Anonymization methods: {anon_methods}")

    lp_detector = SAM3LicensePlateDetector(threshold=args.threshold)

    tasks = [(img, output_dir, anon_methods, lp_detector) for img in images]
    
    for task in tqdm(tasks, desc="Processing"):
        process_image(task)

    print("\nDone!")


if __name__ == "__main__":
    main()
