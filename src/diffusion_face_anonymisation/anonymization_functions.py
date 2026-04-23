import logging
import inspect
from pathlib import Path
from typing import Callable
import numpy as np
from PIL import Image
from skimage.filters import gaussian

from diffusion_face_anonymisation.face import Face, add_face_cutout_and_mask_img
from diffusion_face_anonymisation.body import Body, add_body_cutout_and_mask_img
from diffusion_face_anonymisation.license_plate import LicensePlate
from diffusion_face_anonymisation.io_functions import get_faces_from_file
from diffusion_face_anonymisation.io_functions import get_bodies_from_file
import diffusion_face_anonymisation.utils as utils
from diffusion_face_anonymisation.body_detection import BodyDetector
from diffusion_face_anonymisation.license_plate_detection import LicensePlateDetector


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
        # Get the cutout region and make it white
        cutout = np.array(obj.body_cutout)
        obj.body_anon = Image.fromarray(np.ones_like(cutout) * 255)
    elif isinstance(obj, LicensePlate):
        obj.lp_anon = Image.fromarray(np.ones_like(np.array(obj.lp_cutout)) * 255)
    return obj


def anonymize_gauss(*, obj) -> object:
    if isinstance(obj, Face):
        obj.face_anon = Image.fromarray(
            gaussian(
                np.array(obj.face_cutout, dtype=np.uint8),
                preserve_range=True,
                sigma=3,
                channel_axis=-1,  # type: ignore
            ).astype(np.uint8)
        )
    elif isinstance(obj, Body):
        res = gaussian(
            np.array(obj.body_cutout, dtype=np.uint8),
            preserve_range=True,
            sigma=3,
            channel_axis=-1,  # type: ignore
        ).astype(np.uint8)
        obj.body_anon = Image.fromarray(res)
    elif isinstance(obj, LicensePlate):
        # Scale sigma based on LP size - need stronger blur for larger plates
        # Typical LP is ~100px wide, use sigma = max(width, height) / 8
        lp_img = np.array(obj.lp_cutout, dtype=np.uint8)
        h, w = lp_img.shape[:2]
        sigma = max(w, h) / 8  # e.g., 80px -> sigma=10
        sigma = max(sigma, 5)  # minimum sigma=5 for readability protection
        res = gaussian(
            lp_img,
            preserve_range=True,
            sigma=sigma,
            channel_axis=-1,  # type: ignore
        ).astype(np.uint8)
        obj.lp_anon = Image.fromarray(res)
    return obj


def anonymize_pixelize(*, obj, pixels_per_block=8) -> object:
    if isinstance(obj, Face):
        obj_img = np.array(obj.face_cutout.copy())
    elif isinstance(obj, Body):
        obj_img = np.array(obj.body_cutout.copy())
    elif isinstance(obj, LicensePlate):
        obj_img = np.array(obj.lp_cutout.copy())
        # Scale block size based on LP size - larger blocks for larger plates
        # Goal: ~6-8 blocks across the width to obscure characters
        h, w = obj_img.shape[:2]
        pixels_per_block = max(w // 6, 8)  # e.g., 80px -> 13px blocks
    else:
        return obj

    for idx_v in range(obj_img.shape[0] // pixels_per_block):
        for idx_u in range(obj_img.shape[1] // pixels_per_block):
            block = obj_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ]
            mean = np.mean(
                np.reshape(block, [pixels_per_block * pixels_per_block, 3]), axis=0
            )
            obj_img[
                idx_v * pixels_per_block : (idx_v + 1) * pixels_per_block,
                idx_u * pixels_per_block : (idx_u + 1) * pixels_per_block,
            ] = mean

    if isinstance(obj, Face):
        obj.face_anon = Image.fromarray(obj_img)
    elif isinstance(obj, Body):
        obj.body_anon = Image.fromarray(obj_img)
    elif isinstance(obj, LicensePlate):
        obj.lp_anon = Image.fromarray(obj_img)

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
    face.face_anon = Image.fromarray(
        inpainted_img_np[face.bounding_box.get_slice_area()]
    )

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


# ============================================================================
# Multi-GPU Anonymization Functions
# ============================================================================

def anonymize_face_with_lda_multigpu(
    *,
    face: Face,
    img: Image.Image,
    endpoint_pool,
    image_file: str = "unknown",
    face_index: int = 0
) -> Face:
    """
    Anonymize face using LDA with multi-GPU support.
    
    Args:
        face: Face object with cutout and mask
        img: Mask image for face inpainting
        endpoint_pool: APIEndpointPool for load balancing
        image_file: Source image file (for tracking)
        face_index: Index of face in image (for tracking)
        
    Returns:
        Face object with anonymized result in face.face_anon
    """
    init_img_b64 = utils.encode_image_to_b64(img)
    mask_b64 = utils.encode_image_to_b64(face.mask_image)
    png_payload = utils.fill_face_payload(init_img_b64, mask_b64)
    
    # Send to endpoint pool (automatically load-balanced)
    inpainted_img_b64 = endpoint_pool.send_request(
        png_payload,
        image_file=image_file,
        object_type='face',
        object_index=face_index
    )
    
    inpainted_img = utils.convert_b64_to_pil(inpainted_img_b64)
    inpainted_img_np = np.array(inpainted_img)
    face.face_anon = Image.fromarray(
        inpainted_img_np[face.bounding_box.get_slice_area()]
    )
    
    return face


def anonymize_body_with_lda_multigpu(
    *,
    body: Body,
    img: Image.Image,
    endpoint_pool,
    image_file: str = "unknown",
    body_index: int = 0
) -> Body:
    """
    Anonymize body using LDA with multi-GPU support.
    
    Args:
        body: Body object with cutout and mask
        img: Full image for body inpainting
        endpoint_pool: APIEndpointPool for load balancing
        image_file: Source image file (for tracking)
        body_index: Index of body in image (for tracking)
        
    Returns:
        Body object with anonymized result in body.body_anon
    """
    init_img_b64 = utils.encode_image_to_b64(img)
    mask_b64 = utils.encode_image_to_b64(body.body_mask_image)
    pose_img_b64 = utils.encode_image_to_b64(body.body_cutout)
    png_payload = utils.fill_body_payload(init_img_b64, mask_b64, pose_img_b64)
    
    # Send to endpoint pool (automatically load-balanced)
    inpainted_img_b64 = endpoint_pool.send_request(
        png_payload,
        image_file=image_file,
        object_type='body',
        object_index=body_index
    )
    
    inpainted_img = utils.convert_b64_to_pil(inpainted_img_b64)
    inpainted_img_np = np.array(inpainted_img)
    body.body_anon = Image.fromarray(inpainted_img_np)
    
    return body


def anonymize_face_image(
    image_file: Path, mask_file: Path, anon_function: Callable
) -> Image.Image:
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
        logging.debug(f"Anonymizing body with the following values: {body}")
        if "img" in inspect.signature(anon_function).parameters:
            body = anon_function(obj=body, img=image)
        else:
            body = anon_function(obj=body)

        final_image = body.add_anon_body_to_image(final_image)

    return Image.fromarray(final_image), bodies


def anonymize_body_image_with_cached_bodies(
    image_file: Path,
    bodies: list[Body],
    anon_function: Callable,
) -> tuple[Image.Image, list[Body]]:
    """Like anonymize_body_image but skips detection - uses pre-detected bodies.
    
    This is faster when processing the same image with multiple anonymization methods.
    """
    image = Image.open(image_file)
    final_image = np.array(image)
    bodies = add_body_cutout_and_mask_img(bodies, final_image)
    for body in bodies:
        if "img" in inspect.signature(anon_function).parameters:
            body = anon_function(obj=body, img=image)
        else:
            body = anon_function(obj=body)
        final_image = body.add_anon_body_to_image(final_image)

    return Image.fromarray(final_image), bodies


def anonymize_lp_image(
    image_file: Path,
    anon_function: Callable,
    detector: LicensePlateDetector,
) -> Image.Image:
    """Anonymize license plates in image using detector."""
    image = Image.open(image_file)
    final_image = np.array(image)
    
    license_plates = detector.detect_lp_in_image(image_file)
    logging.debug(f"Found {len(license_plates)} license plates in {Path(image_file).stem}")
    
    for lp in license_plates:
        lp = anon_function(obj=lp)
        final_image = lp.add_anon_lp_to_image(final_image)
    
    return Image.fromarray(final_image)


def anonymize_lp_image_with_cached_lps(
    image_file: Path,
    license_plates: list[LicensePlate],
    anon_function: Callable,
) -> tuple[Image.Image, list[LicensePlate]]:
    """Like anonymize_lp_image but uses pre-detected license plates (faster for multi-method).
    
    This is faster when processing the same image with multiple anonymization methods.
    """
    image = Image.open(image_file)
    final_image = np.array(image)
    
    for lp in license_plates:
        lp = anon_function(obj=lp)
        final_image = lp.add_anon_lp_to_image(final_image)
    
    return Image.fromarray(final_image), license_plates


def anonymize_combined_body_and_lp(
    image_file: Path,
    bodies: list[Body],
    license_plates: list[LicensePlate],
    body_anon_function: Callable,
    lp_anon_function: Callable,
    endpoint_pool=None,
) -> tuple[Image.Image, list[Body], list[LicensePlate]]:
    """Anonymize both bodies and license plates in same image.
    
    Bodies are anonymized first, then license plates composited on top.
    
    Args:
        image_file: Path to input image
        bodies: Pre-detected Body objects
        license_plates: Pre-detected LicensePlate objects
        body_anon_function: Anonymization function for bodies
        lp_anon_function: Anonymization function for license plates
        endpoint_pool: Optional APIEndpointPool for multi-GPU LDA
        
    Returns:
        Tuple of (anonymized_image, bodies, license_plates)
    """
    image = Image.open(image_file)
    final_image = np.array(image)
    
    # Anonymize bodies first
    bodies = add_body_cutout_and_mask_img(bodies, final_image)
    
    # Check if we should use multi-GPU LDA
    body_func_name = getattr(body_anon_function, '__name__', '')
    use_multigpu = endpoint_pool is not None and body_func_name == 'anonymize_lda'
    
    for idx, body in enumerate(bodies):
        if use_multigpu:
            # Use multi-GPU LDA
            body = anonymize_body_with_lda_multigpu(
                body=body, 
                img=image,
                endpoint_pool=endpoint_pool,
                image_file=str(image_file),
                body_index=idx
            )
        elif "img" in inspect.signature(body_anon_function).parameters:
            body = body_anon_function(obj=body, img=image)
        else:
            body = body_anon_function(obj=body)
        final_image = body.add_anon_body_to_image(final_image)
    
    # Then anonymize license plates
    for lp in license_plates:
        lp = lp_anon_function(obj=lp)
        final_image = lp.add_anon_lp_to_image(final_image)
    
    return Image.fromarray(final_image), bodies, license_plates
