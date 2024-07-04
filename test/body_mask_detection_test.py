import copy
import unittest
import os
import numpy as np
from PIL import Image
import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.body_detection import BodyDetector
# we only need this for testing because in deployment we don't need the numpy functionality
def convert_pil_to_np(image):
    return np.array(image)

class TestSingleMaskHandling(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "gtFine"))
    #TODO: init the body detector correctly
    body_detector = BodyDetector()

    def test_(self):
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        # we could skip the dfa_utils in the tests to prevent going pil->numpy->pil but this way we actually test the functions we later use in the scripts
        for i, img_file in enumerate(png_files):
            # get one pedestrian mask with full white pixels
            persons_cutout, persons_white_mask = self.body_detector.detect(img_file)
            for j, (person_cutout, person_white_mask) in enumerate(zip(persons_cutout, persons_white_mask)):
                # apply mask to image
                person_cutout_pil = Image.fromarray(person_cutout)
                person_white_mask_pil = Image.fromarray(person_white_mask)
                person_cutout_pil.save(f"/tmp/dfa_body_cutout_test_{i}_{j}.png")
                person_white_mask_pil.save(f"/tmp/dfa_body_mask_test_{i}_{j}.png")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
