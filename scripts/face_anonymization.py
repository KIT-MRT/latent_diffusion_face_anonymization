from PIL import Image
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import logging

import diffusion_face_anonymisation.utils as dfa_utils
from diffusion_face_anonymisation.io_functions import setup_parser_and_parse_args
from diffusion_face_anonymisation.anonymization_functions import define_anon_function

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

        image = Image.open(image_file)
        debug_img = np.array(image)

        all_faces_bb_list = dfa_utils.get_face_bounding_box_list_from_file(mask_file)
        mask_dict_list = dfa_utils.convert_bb_to_mask_dict_list(
            all_faces_bb_list, image_width=image.width, image_height=image.height
        )
        logger.debug(
            f"Found {len(mask_dict_list)} faces in image {Path(image_file).stem}"
        )

        inpainted_img_list = [
            anon_function(image=image, mask=mask_dict["bb"])  # type: ignore
            for mask_dict in mask_dict_list
        ]

        final_img = dfa_utils.add_inpainted_faces_to_orig_img(
            image, inpainted_img_list, mask_dict_list
        )

        orig_file = Path(image_file)
        # Construct paths for output and debug images
        output_filename = f"{orig_file.stem}_anon_{anon_function}{orig_file.suffix}"
        debug_img_filename = (
            f"debug_{orig_file.stem}_anon_{anon_function}{orig_file.suffix}"
        )

        output_path = output_dir / output_filename
        debug_output_path = output_dir / "debug" / debug_img_filename

        # Save the final image and debug image
        final_img.save(output_path)
        Image.fromarray(debug_img).save(debug_output_path)

        logger.info(f"Anonymized image saved to {output_path}")
        logger.debug(f"Debug image saved to {debug_output_path}")
