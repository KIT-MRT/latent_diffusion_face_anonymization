import unittest
import os
import logging
from pathlib import Path
import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.detection_utils import BodyDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_body_detection.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class BodyDetectionTest(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))

    def test_body_detection(self):
        bd = BodyDetector()
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_files = [
            Path(os.path.join(self.test_image_base_path, img)) for img in png_files
        ]
        bd.body_detect_in_files(image_files=image_files, output_dir=Path("/tmp"))

        logging.info(f"Body detection test completed for {len(image_files)} images")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
