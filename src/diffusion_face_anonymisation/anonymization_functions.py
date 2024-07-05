import numpy as np
from PIL import Image
from pathlib import Path
from skimage.filters import gaussian
from typing import Callable
import logging
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from diffusion_face_anonymisation.utils import FaceBoundingBox
import diffusion_face_anonymisation.utils as dfa_utils


def define_anon_function(anon_method: str) -> Callable:
    anon_functions = {
        "white": anonymize_face_white,
        "gauss": anonymize_face_gauss,
        "pixel": anonymize_face_pixelize,
        "ldfa": anonymize_face_ldfa,  # TODO: implement new ldfa function
    }
    return anon_functions.get(anon_method)  # type: ignore


def anonymize_face_ldfa(*, image: np.ndarray, mask: FaceBoundingBox):

    # Load the controlnet and the stable diffusion pipeline
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet)

    # Extract the face area
    img_anon = np.array(image)
    face_area = img_anon[mask.get_slice_area()]

    # Generate the anonymized face using the LDFA model
    anonymized_face = pipeline(face_area, prompt="anonymize face").images[0]

    # Replace the original face area with the anonymized face
    img_anon[mask.get_slice_area()] = np.array(anonymized_face)

    return Image.fromarray(img_anon)


def anonymize_face_white(*, image: np.ndarray, mask: FaceBoundingBox):
    img_anon = np.array(image)
    img_anon[mask.get_slice_area()] = [255, 255, 255]
    return Image.fromarray(img_anon)


def anonymize_face_gauss(*, image: np.ndarray, mask: FaceBoundingBox):
    img_anon = np.array(image, dtype=float) / 255
    face_area = img_anon[mask.get_slice_area()]
    img_anon[mask.get_slice_area()] = gaussian(face_area, sigma=3, channel_axis=-1)  # type: ignore
    return Image.fromarray((img_anon * 255).astype(np.uint8))


def anonymize_face_pixelize(*, image, pixels_per_block=8):
    img_anon = np.array(image)

    for idx_v in range(img_anon.shape[0] // pixels_per_block):
        for idx_u in range(img_anon.shape[1] // pixels_per_block):
            block = img_anon[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ]
            mean = np.mean(
                np.reshape(block, [pixels_per_block * pixels_per_block, 3]), axis=0
            )
            img_anon[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ] = mean

    return Image.fromarray(img_anon)


def anonymize_image(image_file: Path, mask_file: Path, anon_function: Callable):

    image = Image.open(image_file)

    all_faces_bb_list = dfa_utils.get_face_bounding_box_list_from_file(mask_file)
    mask_dict_list = dfa_utils.convert_bb_to_mask_dict_list(
        all_faces_bb_list, image_width=image.width, image_height=image.height
    )
    logging.debug(f"Found {len(mask_dict_list)} faces in image {Path(image_file).stem}")

    inpainted_img_list = [
        anon_function(image=image, mask=mask_dict["bb"])  # type: ignore
        for mask_dict in mask_dict_list
    ]

    final_img = dfa_utils.add_inpainted_faces_to_orig_img(
        image, inpainted_img_list, mask_dict_list
    )
    return final_img