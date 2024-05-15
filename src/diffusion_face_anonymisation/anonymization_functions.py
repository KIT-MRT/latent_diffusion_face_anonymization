import numpy as np
from PIL import Image
from skimage.filters import gaussian

from diffusion_face_anonymisation.utils import FaceBoundingBox


def anonymize_face_white(*, image: np.ndarray, mask: FaceBoundingBox):
    img_anon = np.array(image)
    img_anon[mask.get_slice_area()] = [255, 255, 255]
    return Image.fromarray(img_anon)


def anonymize_face_gauss(*, image: np.ndarray, mask: FaceBoundingBox):
    img_anon = np.array(image, dtype=float) / 255
    face_area = img_anon[mask.get_slice_area()]
    img_anon[mask.get_slice_area()] = gaussian(face_area, sigma=3, channel_axis=-1)
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
