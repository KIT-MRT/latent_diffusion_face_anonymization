import os
from tqdm import tqdm
import logging

import diffusion_face_anonymisation.utils as dfa_utils
from diffusion_face_anonymisation.io_functions import setup_parser_and_parse_args
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_image,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_anon.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    image_dir, mask_dir, output_dir, anon_method = setup_parser_and_parse_args()
    anon_function = define_anon_function(anon_method)

    logger.info(
        f"Starting face anonymization with images from {image_dir}, masks from {mask_dir}, output to {output_dir}"
    )
    debug_dir = os.path.join(output_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        logger.info(f"Debug directory created at {debug_dir}")

    image_mask_dict = dfa_utils.get_image_mask_dict(str(image_dir), str(mask_dir))
    logger.info(f"Found {len(image_mask_dict)} images with corresponding masks.")
    logger.info("Starting to anonymize faces in images.")

    for entry in tqdm(image_mask_dict.values()):
        image_file = entry["image_file"]
        mask_file = entry["mask_file"]
        logger.info(f"Processing image {image_file} with mask {mask_file}")
        anonymize_image(image_file, mask_file, anon_function)
