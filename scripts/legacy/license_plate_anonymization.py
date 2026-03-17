"""License Plate Anonymization Script

Usage:
    python license_plate_anonymization.py --image_dir data/ --output_dir output/ --method white
    python license_plate_anonymization.py --image_dir data/ --output_dir output/ --method all
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_lp_image_with_cached_lps,
)
from diffusion_face_anonymisation.io_functions import save_anon_image


def get_image_files(image_dir: Path, ext: str = "png") -> list[Path]:
    """Get all images from directory."""
    return sorted(image_dir.glob(f"**/*.{ext}"))


def main():
    parser = argparse.ArgumentParser(description="Anonymize license plates in images")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["white", "gauss", "pixel", "all"])
    parser.add_argument("--ext", type=str, default="png")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["white", "gauss", "pixel"] if args.method == "all" else [args.method]
    images = get_image_files(image_dir, args.ext)

    print(f"Processing {len(images)} images with methods: {methods}")

    detector = LicensePlateDetector(conf_threshold=args.conf)

    for img_path in tqdm(images, desc="Anonymizing"):
        plates = detector.detect(img_path)

        for method in methods:
            anon_func = define_anon_function(method)
            anon_img, _ = anonymize_lp_image_with_cached_lps(img_path, plates, anon_func)
            save_anon_image(anon_img, str(img_path), output_dir, f"lp_{method}")

    print(f"Done! Output in {output_dir}")


if __name__ == "__main__":
    main()
