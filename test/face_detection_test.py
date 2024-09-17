import unittest
import os
import logging

import diffusion_face_anonymisation.io_functions as dfa_io
import diffusion_face_anonymisation.utils as dfa_utils
from diffusion_face_anonymisation.detection_utils import detect_faces_in_files


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_face_detection.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class FaceInpaintingTest(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))

    def test_(self):
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(
            png_files, image_mask_dict, "image_file"
        )
        image_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        detect_faces_in_files(image_files, self.test_image_base_path, output_dir="/tmp")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
