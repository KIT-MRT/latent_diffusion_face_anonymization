import copy
import unittest
import os
import numpy as np
from PIL import Image
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io
# we only need this for testing because in deployment we don't need the numpy functionality
def convert_pil_to_np(image):
    return np.array(image)

class TestSingleMaskHandling(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "gtFine"))

    def test_(self):
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        sem_mask_png_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "labelIds.png")
        inst_mask_png_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "instanceIds.png")
        assert(len(png_files) == len(sem_mask_png_files))
        assert(len(sem_mask_png_files) == len(inst_mask_png_files))
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_path_to_body_mask_dict(sem_mask_png_files, image_mask_dict, "label_ids_file")
        image_mask_dict = dfa_utils.add_file_path_to_body_mask_dict(inst_mask_png_files, image_mask_dict, "instance_ids_file")
        image_mask_dict = dfa_utils.add_file_path_to_body_mask_dict(png_files, image_mask_dict, "image_file")
        # we could skip the dfa_utils in the tests to prevent going pil->numpy->pil but this way we actually test the functions we later use in the scripts
        for i, entry in enumerate(image_mask_dict.values()):
            # get one pedestrian mask with full white pixels
            persons_cutout, persons_white_mask = dfa_utils.get_persons_cutout_and_mask(entry)
            for j, (person_cutout, person_white_mask) in enumerate(zip(persons_cutout, persons_white_mask)):
                # apply mask to image
                person_cutout_pil = Image.fromarray(person_cutout)
                person_white_mask_pil = Image.fromarray(person_white_mask)
                person_cutout_pil.save(f"/tmp/dfa_body_cutout_test_{i}_{j}.png")
                person_white_mask_pil.save(f"/tmp/dfa_body_mask_test_{i}_{j}.png")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
