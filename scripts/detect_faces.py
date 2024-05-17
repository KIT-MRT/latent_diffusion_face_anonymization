import os
import logging

from diffusion_face_anonymisation.detection_utils import detect_faces_in_files
import diffusion_face_anonymisation.io_functions as dfa_io


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_detection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    image_dir, output_dir = dfa_io.setup_face_detection_parser_and_parse_args()

    if not os.path.exists(image_dir):
        logger.error(f"Input directory does not exist: {image_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Output directory created: {output_dir}")

    image_files = dfa_io.glob_files_by_extension(str(image_dir), "png")
    detect_faces_in_files(image_files, image_dir, output_dir)
