"""Simple test for combined body + license plate anonymization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/tmp/YOLOX")

from diffusion_face_anonymisation.body_detection import BodyDetector
from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_combined_body_and_lp,
)


def test_combined():
    """Test combined body + LP anonymization."""
    print("Testing combined body + LP anonymization...")

    body_detector = BodyDetector()
    lp_detector = LicensePlateDetector()

    test_dir = Path(__file__).parent.parent / "data"
    img = next(test_dir.glob("*.png"))

    bodies = body_detector.body_detect_in_image(img)
    plates = lp_detector.detect(img)

    print(f"  Image: {img.name}")
    print(f"  Bodies: {len(bodies)}, Plates: {len(plates)}")

    body_func = define_anon_function("white")
    lp_func = define_anon_function("pixel")

    anon_img, _, _ = anonymize_combined_body_and_lp(img, bodies, plates, body_func, lp_func)
    print(f"  Output size: {anon_img.size}")

    print("OK")


if __name__ == "__main__":
    test_combined()
    print("\nAll tests passed!")
