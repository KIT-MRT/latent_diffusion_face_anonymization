import unittest
import os
import logging
from pathlib import Path
from diffusion_face_anonymisation.face import add_face_cutout_and_mask_img
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_ldfa.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class TestSingleMaskHandling(unittest.TestCase):
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

        for i, entry in enumerate(image_mask_dict.values()):
            image = dfa_utils.preprocess_image(entry["image_file"])
            faces = dfa_io.get_faces_from_file(entry["mask_file"])
            faces = add_face_cutout_and_mask_img(faces, image)
            for j, face in enumerate(faces):
                face.save(Path("/storage_local/roesch/ldba/faces_tmp"), i, j)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
