import logging
import inspect
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage.filters import gaussian
import io
import base64

from diffusion_face_anonymisation.face import (
    Face,
    add_face_cutout_and_mask_img,
)
from diffusion_face_anonymisation.io_functions import get_faces_from_file
from diffusion_face_anonymisation.utils import (
    encode_image_mask_to_b64,
    fill_png_payload,
    send_request_to_api,
)


def define_anon_function(anon_method: str) -> Callable:
    anon_functions = {
        "white": anonymize_face_white,
        "gauss": anonymize_face_gauss,
        "pixel": anonymize_face_pixelize,
        "ldfa": anonymize_face_ldfa,  # TODO: implement new ldfa function
    }
    return anon_functions.get(anon_method)  # type: ignore


def anonymize_face_ldfa(*, face: Face, img:Image.Image) -> Face:
    init_img_b64, mask_b64 = encode_image_mask_to_b64(img, face.mask_image)
    png_payload = fill_png_payload(init_img_b64, mask_b64)
    inpainted_img_b64 = send_request_to_api(png_payload)

    def convert_b64_to_pil(img_b64):
        return Image.open(io.BytesIO(base64.b64decode(img_b64.split(",", 1)[0])))

    inpainted_img = convert_b64_to_pil(inpainted_img_b64)
    inpainted_img_np = np.array(inpainted_img)
    face.face_anon = Image.fromarray(inpainted_img_np[face.bounding_box.get_slice_area()])

    return face


def anonymize_face_white(*, face: Face) -> Face:
    face.face_anon = Image.fromarray(np.ones_like(np.array(face.face_cutout)) * 255)
    return face


def anonymize_face_gauss(*, face: Face) -> Face:
    face.face_anon = gaussian(
        np.array(face.face_cutout),
        sigma=3,
        channel_axis=-1,  # type: ignore
    )
    return face


def anonymize_face_pixelize(*, face: Face) -> Face:
    pixels_per_block = 8
    face_img = np.array(face.face_cutout.copy())

    for idx_v in range(face_img.shape[0] // pixels_per_block):
        for idx_u in range(face_img.shape[1] // pixels_per_block):
            block = face_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ]
            mean = np.mean(
                np.reshape(block, [pixels_per_block * pixels_per_block, 3]), axis=0
            )
            face_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ] = mean
    face.face_anon = Image.fromarray(face_img)
    return face


def anonymize_image(image_file: Path, mask_file: Path, anon_function: Callable) -> Image.Image:
    image = Image.open(image_file)

    faces = get_faces_from_file(mask_file)
    faces = add_face_cutout_and_mask_img(faces=faces, image=np.array(image))
    logging.debug(f"Found {len(faces)} faces in image {Path(image_file).stem}")
    final_img = np.array(image)

    for face in faces:
        # check if the function has an img parameter
        if 'img' in inspect.signature(anon_function).parameters:
            face = anon_function(face=face, img=image)
        else:
            face = anon_function(face=face)
        final_img = face.add_anon_face_to_image(final_img)

    return Image.fromarray(final_img)
