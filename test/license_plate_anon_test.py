"""Simple test for license plate detection and anonymization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/tmp/YOLOX")

from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_lp_image_with_cached_lps,
)


def test_detection():
    """Test license plate detection."""
    print("Testing LP detection...")
    detector = LicensePlateDetector()

    test_dir = Path(__file__).parent.parent / "data"
    for img in test_dir.glob("*.png"):
        plates = detector.detect(img)
        print(f"  {img.name}: {len(plates)} plates")
        for lp in plates:
            print(f"    - bbox={lp.aabb}")

    print("OK")


def test_anonymization():
    """Test license plate anonymization."""
    print("Testing LP anonymization...")
    detector = LicensePlateDetector()

    test_dir = Path(__file__).parent.parent / "data"
    img = next(test_dir.glob("*.png"))
    plates = detector.detect(img)

    for method in ["white", "gauss", "pixel"]:
        anon_func = define_anon_function(method)
        anon_img, _ = anonymize_lp_image_with_cached_lps(img, plates, anon_func)
        print(f"  {method}: {anon_img.size}")

    print("OK")


if __name__ == "__main__":
    test_detection()
    test_anonymization()
    print("\nAll tests passed!")
