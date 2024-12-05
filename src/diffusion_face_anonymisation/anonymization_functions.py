import logging
import inspect
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from skimage.filters import gaussian

from diffusion_face_anonymisation.face import Face, add_face_cutout_and_mask_img
from diffusion_face_anonymisation.body import Body, add_body_cutout_and_mask_img
from diffusion_face_anonymisation.io_functions import get_faces_from_file
from diffusion_face_anonymisation.io_functions import get_bodies_from_file
import diffusion_face_anonymisation.utils as utils
from diffusion_face_anonymisation.body_detection import BodyDetector


def define_anon_function(anon_method: str):
    anon_functions = {
        "white": anonymize_white,
        "gauss": anonymize_gauss,
        "pixel": anonymize_pixelize,
        "lda": anonymize_lda,
    }
    return anon_functions.get(anon_method)


def anonymize_white(*, obj) -> object:
    if isinstance(obj, Face):
        obj.face_anon = Image.fromarray(np.ones_like(np.array(obj.face_cutout)) * 255)
    elif isinstance(obj, Body):
        obj.body_anon = obj.body_mask
    return obj


def anonymize_gauss(*, obj) -> object:
    if isinstance(obj, Face):
        obj.face_anon = gaussian(
            np.array(obj.face_cutout, dtype=np.uint8),
            preserve_range=True,
            sigma=3,
            channel_axis=-1,  # type: ignore
        )
    elif isinstance(obj, Body):
        obj.body_anon = gaussian(
            np.array(obj.body_cutout, dtype=np.uint8),
            preserve_range=True,
            sigma=3,
            channel_axis=-1,  # type: ignore
        )
    return obj


def anonymize_pixelize(*, obj, pixels_per_block=8) -> object:
    if isinstance(obj, Face):
        obj_img = np.array(obj.face_cutout.copy())
    elif isinstance(obj, Body):
        obj_img = np.array(obj.body_cutout.copy())

    for idx_v in range(obj_img.shape[0] // pixels_per_block):
        for idx_u in range(obj_img.shape[1] // pixels_per_block):
            block = obj_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ]
            mean = np.mean(np.reshape(block, [pixels_per_block * pixels_per_block, 3]), axis=0)
            obj_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ] = mean

    if isinstance(obj, Face):
        obj.face_anon = Image.fromarray(obj_img)
    elif isinstance(obj, Body):
        obj.body_anon = Image.fromarray(obj_img)

    return obj


def anonymize_lda(*, obj, img: Image.Image) -> object:
    if isinstance(obj, Face):
        obj = anonymize_face_with_lda(face=obj, img=obj.mask_image)
    elif isinstance(obj, Body):
        obj = anonymize_body_with_lda(body=obj, img=img)
    return obj


def anonymize_face_with_lda(*, face: Face, img: Image.Image) -> Face:
    init_img_b64 = utils.encode_image_to_b64(img)
    mask_b64 = utils.encode_image_to_b64(face.mask_image)
    png_payload = utils.fill_face_payload(init_img_b64, mask_b64)
    inpainted_img_b64 = utils.send_request_to_api(png_payload)

    inpainted_img = utils.convert_b64_to_pil(inpainted_img_b64)
    inpainted_img_np = np.array(inpainted_img)
    face.face_anon = Image.fromarray(inpainted_img_np[face.bounding_box.get_slice_area()])

    return face


def anonymize_body_with_lda(*, body: Body, img: Image.Image) -> Body:
    init_img_b64 = utils.encode_image_to_b64(img)
    mask_b64 = utils.encode_image_to_b64(body.body_mask_image)
    pose_img_b64 = utils.encode_image_to_b64(body.body_cutout)
    png_payload = utils.fill_body_payload(init_img_b64, mask_b64, pose_img_b64)
    inpainted_img_b64 = utils.send_request_to_api(png_payload)
    inpainted_img = utils.convert_b64_to_pil(inpainted_img_b64)
    inpainted_img_np = np.array(inpainted_img)
    body.body_anon = Image.fromarray(inpainted_img_np)
    return body


def anonymize_face_image(image_file: Path, mask_file: Path, anon_function: Callable) -> Image.Image:
    image = Image.open(image_file)
    final_image = np.array(image)
    faces = get_faces_from_file(mask_file)
    faces = add_face_cutout_and_mask_img(faces=faces, image=np.array(image))
    logging.debug(f"Found {len(faces)} faces in image {Path(image_file).stem}")

    for face in faces:
        if "img" in inspect.signature(anon_function).parameters:
            face = anon_function(obj=face, img=image)
        else:
            face = anon_function(obj=face)
        final_image = face.add_anon_face_to_image(final_image)

    return Image.fromarray(final_image)


def anonymize_body_image(
    image_file: Path,
    mask_files: dict[str, str],
    anon_function: Callable,
    detector: BodyDetector | None,
) -> tuple[Image.Image, list[Body]]:
    image = Image.open(image_file)
    final_image = np.array(image)
    # get bodies via the body detector
    if detector:
        bodies = detector.body_detect_in_image(image_file)
    else:
        bodies = get_bodies_from_file(mask_files)
    logging.debug(f"Found {len(bodies)} bodies in image {Path(image_file).stem}")
    bodies = add_body_cutout_and_mask_img(bodies, final_image)
    for body in bodies:
        if "img" in inspect.signature(anon_function).parameters:
            body = anon_function(obj=body, img=image)
        else:
            body = anon_function(obj=body)

        final_image = body.add_anon_body_to_image(final_image)

    return Image.fromarray(final_image), bodies
