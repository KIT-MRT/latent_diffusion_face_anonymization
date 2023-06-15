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
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "annotations"))

    def test_(self):
        json_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "json")
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(json_files, image_mask_dict, "mask_file")
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(png_files, image_mask_dict, "image_file")
        # we could skip the dfa_utils in the tests to prevent going pil->numpy->pil but this way we actually test the functions we later use in the scripts
        for i, entry in enumerate(image_mask_dict.values()):
            image = dfa_utils.preprocess_image(entry["image_file"])
            image = convert_pil_to_np(image)
            all_faces_bb_list = dfa_utils.get_face_bounding_box_list_from_file(entry["mask_file"])

            mask_image_list = dfa_utils.convert_bb_to_mask_image_list(all_faces_bb_list, image.shape[1], image.shape[0])
            for j, mask_image_dict in enumerate(mask_image_list):
                # apply mask to image
                mask_area = np.where(mask_image_dict["mask"] == 255)
                image_mask = copy.deepcopy(image)
                image_mask[:,:,:] = (0,0,0)
                image_mask[mask_area] = (255, 255, 255)
                image_pil = Image.fromarray(image_mask)
                image_pil.save(f"/tmp/dfa_mask_test_{i}_{j}.png")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
