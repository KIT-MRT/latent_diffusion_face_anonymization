from abc import ABC
import unittest
import os
import logging
from pathlib import Path
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_body_image,
)
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_simple_body_anon.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class BaseTestBodyAnon(unittest.TestCase, ABC):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "gtFine"))
    output_dir = Path("/tmp/ldfa_tests")
    os.makedirs(output_dir, exist_ok=True)
    anon_type = None
    image_mask_dict = dfa_utils.get_image_mask_dict(
        test_image_base_path, test_mask_base_path, method="body"
    )

    def setUp(self):
        assert self.anon_type is not None
        self.anon_function = define_anon_function(self.anon_type)

    def run_test(self):
        assert self.anon_type is not None
        assert self.anon_function is not None

        for entry in self.image_mask_dict.values():
            image_file = entry["image_file"]
            logger.info(f"Processing image {image_file} with mask {entry['label_ids_file']}")
            anon_img = anonymize_body_image(image_file, entry, self.anon_function)
            dfa_io.save_anon_image(anon_img, image_file, self.output_dir, self.anon_type)


class TestBodyMaskAnon(BaseTestBodyAnon):
    anon_type = "white"

    def test_mask_anon(self):
        self.run_test()


class TestBodyPixelAnon(BaseTestBodyAnon):
    anon_type = "pixel"

    def test_pixel_anon(self):
        self.run_test()


class TestBodyGaussAnon(BaseTestBodyAnon):
    anon_type = "gauss"

    def test_gauss_anon(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
