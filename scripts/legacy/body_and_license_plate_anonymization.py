"""Combined Body + License Plate Anonymization Script

Usage:
    # Single combination
    python body_and_license_plate_anonymization.py \\
        --image_dir data/ --output_dir output/ --body_method white --lp_method pixel

    # All 12 combinations (4 body x 3 lp methods)
    python body_and_license_plate_anonymization.py \\
        --image_dir data/ --output_dir output/ --all_combinations
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from diffusion_face_anonymisation.body_detection import BodyDetector
from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_combined_body_and_lp,
)


def get_image_files(image_dir: Path, ext: str = "png") -> list[Path]:
    """Get all images from directory."""
    return sorted(image_dir.glob(f"**/*.{ext}"))


def main():
    parser = argparse.ArgumentParser(description="Anonymize bodies and license plates")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--body_method", type=str, choices=["white", "gauss", "pixel", "lda"])
    parser.add_argument("--lp_method", type=str, choices=["white", "gauss", "pixel"])
    parser.add_argument("--all_combinations", action="store_true")
    parser.add_argument("--ext", type=str, default="png")
    parser.add_argument("--lp_conf", type=float, default=0.25)
    args = parser.parse_args()

    if not args.all_combinations and (not args.body_method or not args.lp_method):
        parser.error("Specify --body_method and --lp_method, or use --all_combinations")

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine method combinations
    if args.all_combinations:
        body_methods = ["white", "gauss", "pixel", "lda"]
        lp_methods = ["white", "gauss", "pixel"]
    else:
        body_methods = [args.body_method]
        lp_methods = [args.lp_method]

    images = get_image_files(image_dir, args.ext)
    total_outputs = len(images) * len(body_methods) * len(lp_methods)

    print(f"Processing {len(images)} images")
    print(f"Body methods: {body_methods}")
    print(f"LP methods: {lp_methods}")
    print(f"Total outputs: {total_outputs}")

    # Initialize detectors
    body_detector = BodyDetector()
    lp_detector = LicensePlateDetector(conf_threshold=args.lp_conf)

    pbar = tqdm(total=total_outputs, desc="Anonymizing")

    for img_path in images:
        # Detect once per image
        bodies = body_detector.body_detect_in_image(img_path)
        plates = lp_detector.detect(img_path)

        for body_method in body_methods:
            for lp_method in lp_methods:
                body_func = define_anon_function(body_method)
                lp_func = define_anon_function(lp_method)

                anon_img, _, _ = anonymize_combined_body_and_lp(
                    img_path, bodies, plates, body_func, lp_func
                )

                out_name = f"{img_path.stem}_anon_body_{body_method}_lp_{lp_method}.png"
                anon_img.save(output_dir / out_name)
                pbar.update(1)

    pbar.close()
    print(f"Done! Output in {output_dir}")


if __name__ == "__main__":
    main()
