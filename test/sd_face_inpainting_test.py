import copy
import unittest
import os

import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io

from PIL import Image

class FaceInpaintingTest(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "annotations"))

    def test_(self):
        json_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "json")
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(json_files, image_mask_dict, "mask_file")
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(png_files, image_mask_dict, "image_file")

        for i, entry in enumerate(image_mask_dict.values()):
            image = Image.open(entry["image_file"])
            all_faces_bb_list = dfa_utils.get_face_bounding_box_list_from_file(entry["mask_file"])
            mask_dict_list = dfa_utils.convert_bb_to_mask_dict_list(all_faces_bb_list, image_width=image.width, image_height=image.height)
            inpainted_img_list = []
            for j, mask_dict in enumerate(mask_dict_list):
                mask = Image.fromarray(mask_dict["mask"])
                inpainted_img_list.append(dfa_utils.request_inpaint(init_img=image, mask=mask))
            final_img = dfa_utils.add_inpainted_faces_to_orig_img(image, inpainted_img_list, mask_dict_list)
            final_img.save(f"/tmp/all_faces_inpainted_{i}.png")

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")

