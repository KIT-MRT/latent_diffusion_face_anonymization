import logging
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from diffusion_face_anonymisation.face import (
    Face,
    add_face_cutout_and_mask_img,
)
from diffusion_face_anonymisation.io_functions import get_faces_from_file


def define_anon_function(anon_method: str) -> Callable:
    anon_functions = {
        "white": anonymize_face_white,
        "gauss": anonymize_face_gauss,
        "pixel": anonymize_face_pixelize,
        "ldfa": anonymize_face_ldfa,  # TODO: implement new ldfa function
    }
    return anon_functions.get(anon_method)  # type: ignore


def anonymize_face_ldfa(*, face: Face):
    pass


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


def anonymize_image(image_file: Path, mask_file: Path, anon_function: Callable):
    image = Image.open(image_file)

    faces = get_faces_from_file(mask_file)
    faces = add_face_cutout_and_mask_img(faces=faces, image=np.array(image))
    logging.debug(f"Found {len(faces)} faces in image {Path(image_file).stem}")
    final_img = np.array(image)

    for face in faces:
        face = anon_function(face=face)
        final_img = face.add_anon_face_to_image(final_img)

    return Image.fromarray(final_img)

#Body anonymization Functions

from diffusion_face_anonymisation.body import Body, add_body_cutout_and_mask_img
from diffusion_face_anonymisation.io_functions import get_bodies_from_file


def define_body_anon_function(anon_method: str) -> Callable:
    anon_functions = {
        "white": anonymize_body_white,
        "gauss": anonymize_body_gauss,
        "pixel": anonymize_body_pixelize,
        "ldfa": anonymize_body_ldfa,
    }
    return anon_functions.get(anon_method)


def anonymize_body_white(*, body: Body) -> Body:
    body.body_anon = Image.fromarray(np.ones_like(np.array(body.body_cutout)) * 255)
    return body


def anonymize_body_gauss(*, body: Body) -> Body:
    body.body_anon = Image.fromarray(
        (gaussian(np.array(body.body_cutout), sigma=3, channel_axis=-1) * 255).astype(np.uint8)
    )
    return body


def anonymize_body_pixelize(*, body: Body) -> Body:
    pixels_per_block = 8
    body_img = np.array(body.body_cutout.copy())

    for idx_v in range(body_img.shape[0] // pixels_per_block):
        for idx_u in range(body_img.shape[1] // pixels_per_block):
            block = body_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ]
            mean = np.mean(np.reshape(block, [pixels_per_block * pixels_per_block, 3]), axis=0)
            body_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ] = mean

    body.body_anon = Image.fromarray(body_img)
    return body


def anonymize_body_ldfa(*, body: Body) -> Body:
    return body



def anonymize_body_image(image_file: Path, mask_file: Path, anon_function: Callable) -> Image.Image:
    image = Image.open(image_file)
    final_image = np.array(image)
    bodies = get_bodies_from_file(mask_file)
    logging.debug(f"Body masks loaded from {mask_file}, total bodies found: {len(bodies)}")
    bodies = add_body_cutout_and_mask_img(bodies, final_image)

    for body in bodies:

        body = anon_function(body=body)
        final_image = body.add_anon_body_to_image(final_image)  

    return Image.fromarray(final_image)


