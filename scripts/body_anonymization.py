import os
from tqdm import tqdm
import logging
from pathlib import Path

import diffusion_face_anonymisation.utils as dfa_utils
from diffusion_face_anonymisation.io_functions import (
    setup_parser_and_parse_args,
    save_anon_image,
)
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_body_image,
)
from diffusion_face_anonymisation.body_detection import BodyDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("body_anon.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    image_dir, mask_dir, output_dir, anon_method, image_extension, detect = setup_parser_and_parse_args()
    detector = None
    # load detection model here if no mask dir is provided to avoid reloading of the model
    if detect:
        logger.info("No mask dir provided, using detection model to generate masks.")
        detector = BodyDetector()
    anon_function = define_anon_function(anon_method)
    assert anon_function is not None

    logger.info(
        f"Starting body anonymization with images from {image_dir}, masks from {'detection' if detect else mask_dir}, output to {output_dir}"
    )
    debug_dir = Path(os.path.join(output_dir, "debug"))
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        logger.info(f"Debug directory created at {debug_dir}")

    image_mask_dict = dfa_utils.get_image_mask_dict(str(image_dir), str(mask_dir), method="body", image_file_extension=image_extension, detect=detect)
    if not detect:
        logger.info(f"Found {len(image_mask_dict)} images with corresponding masks.")
    logger.info("Starting to anonymize persons in images.")

    print(f"Found {len(image_mask_dict)} images with corresponding masks.")
    for img_id, entry in tqdm(enumerate(image_mask_dict.values())):
        print(entry)
        image_file = entry["image_file"]
        anon_img, bodies = anonymize_body_image(image_file, entry, anon_function, detector)
        for i, body in enumerate(bodies):
            body.save(debug_dir, img_id, i)
        save_anon_image(anon_img, image_file, output_dir, anon_method)
