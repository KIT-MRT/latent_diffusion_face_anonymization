import unittest
import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from diffusion_face_anonymisation.body import add_body_cutout_and_mask_img
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io


class TestSingleMaskHandling(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "mask_files"))

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
        for i, entry in enumerate(image_mask_dict.values()):
            image = dfa_utils.preprocess_image(entry["image_file"])
            bodies = dfa_io.get_bodies_from_file(entry["mask_file"])
            bodies = add_body_cutout_and_mask_img(bodies, image)
            for j, body in enumerate(bodies):
                body.save(Path("/temp"), i, j)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
