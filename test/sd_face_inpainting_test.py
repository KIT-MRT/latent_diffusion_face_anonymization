import unittest
import os
from pathlib import Path

import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.anonymization_functions import (
    anonymize_image,
    define_anon_function,
)


class FaceInpaintingTest(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "annotations"))

    def test_(self):
        json_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "json")
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(
            json_files, image_mask_dict, "mask_file"
        )
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(
            png_files, image_mask_dict, "image_file"
        )
        anon_function = define_anon_function("ldfa")

        for entry in image_mask_dict.values():
            image_file = entry["image_file"]
            mask_file = entry["mask_file"]
            anon_img = anonymize_image(image_file, mask_file, anon_function)
            dfa_io.save_anon_image(anon_img, image_file, Path("/tmp"), "ldfa")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
