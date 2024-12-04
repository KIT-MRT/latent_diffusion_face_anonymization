import unittest
import os
import logging
from pathlib import Path
from diffusion_face_anonymisation.body import add_body_cutout_and_mask_img
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_body_detection.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class TestSingleMaskHandling(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "gtFine"))
    image_mask_dict = dfa_utils.get_image_mask_dict(
        test_image_base_path, test_mask_base_path, method="body"
    )
    output_dir = Path("/tmp/ldfa_tests")

    def test_(self):
        for i, entry in enumerate(self.image_mask_dict.values()):
            image = dfa_utils.preprocess_image(entry["image_file"])
            bodies = dfa_io.get_bodies_from_file(entry)
            bodies = add_body_cutout_and_mask_img(bodies, image)
            for j, body in enumerate(bodies):
                body.save(self.output_dir, i, j)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
