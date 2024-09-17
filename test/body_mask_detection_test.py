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
    # Get the directory of the current script file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the paths for the test images and masks relative to the script location
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "gtFine"))
    save_base_path = os.path.abspath(os.path.join(file_dir, "Saves_body_cutout"))

    # Initialize the BodyDetector
    body_detector = BodyDetector()

    def test(self):
        # Verify if the test image path exists
        if not os.path.exists(self.test_image_base_path):
            print(f"Directory '{self.test_image_base_path}' does not exist. Skipping test.")
            return

        # Ensure the save base path exists; create it if necessary
        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)

        # Get all PNG files from the test image path
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")

        for i, img_file in enumerate(png_files):
            img_path = os.path.join(self.test_image_base_path, img_file)
            
            # Detect persons and their masks in the image
            persons_cutout, persons_white_mask = self.body_detector.detect(img_file)
            for j, (person_cutout, person_white_mask) in enumerate(zip(persons_cutout, persons_white_mask)):
                # Convert the numpy arrays to PIL images
                person_cutout_pil = Image.fromarray(person_cutout)
                person_white_mask_pil = Image.fromarray((person_white_mask * 255).astype(np.uint8))

                # Define the filenames for saving the images
                cutout_filename = os.path.join(self.save_base_path, f"dfa_body_cutout_test_{i}_{j}.png")
                mask_filename = os.path.join(self.save_base_path, f"dfa_body_mask_test_{i}_{j}.png")

                # Save the cutout and mask images
                person_cutout_pil.save(cutout_filename)
                person_white_mask_pil.save(mask_filename)

                print(f"Saved cutout: {cutout_filename}")
                print(f"Saved mask: {mask_filename}")

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
