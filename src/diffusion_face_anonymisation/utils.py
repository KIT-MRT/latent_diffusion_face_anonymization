from PIL import Image
import numpy as np
import re
from pathlib import Path

import diffusion_face_anonymisation.io_functions as dfa_io

PERSON_LABEL_ID = 24
RIDER_LABEL_ID = 25


def get_image_id(full_image_string):
    match = re.search(r"\w+_\d+_\d+", full_image_string)
    return match.group(0)


def get_image_mask_dict(image_dir: str, mask_dir: str) -> dict:
    png_files = dfa_io.glob_files_by_extension(image_dir, "png")
    json_files = dfa_io.glob_files_by_extension(mask_dir, "json")

    image_mask_dict = {}
    image_mask_dict = add_file_paths_to_image_mask_dict(
        json_files, image_mask_dict, "mask_file"
    )
    image_mask_dict = add_file_paths_to_image_mask_dict(
        png_files, image_mask_dict, "image_file"
    )
    # clear image_mask_dict from entries that do not contain a mask
    image_mask_dict = {
        entry: image_mask_dict[entry]
        for entry in image_mask_dict
        if "mask_file" in image_mask_dict[entry]
    }
    return image_mask_dict


def preprocess_image(path_to_image: str) -> np.ndarray:
    image = Image.open(path_to_image)
    return np.array(image)


def add_file_paths_to_image_mask_dict(
    file_paths: list[Path], image_mask_dict: dict, file_key: str
) -> dict:
    for file in file_paths:
        image_name = file.stem
        image_mask_dict.setdefault(image_name, {})[file_key] = file
    return image_mask_dict


# TODO: merge this with function above
def add_file_path_to_body_mask_dict(file_paths, body_mask_dict, file_key):
    for file in file_paths:
        image_name = file.stem
        image_id = get_image_id(image_name)
        try:
            body_mask_dict[image_id][file_key] = file
        except KeyError:
            body_mask_dict[image_id] = {}
            body_mask_dict[image_id][file_key] = file
    return body_mask_dict


def get_persons_cutout_and_mask(img_dict):
    image = preprocess_image(img_dict["image_file"])
    inst_image = preprocess_image(img_dict["instance_ids_file"])
    if "label_ids_file" in img_dict:
        label_img = preprocess_image(img_dict["label_ids_file"])
        unique_person_pixel_list = get_unique_person_pixel_as_list(
            inst_image, label_img
        )
        persons_white_mask_list = get_persons_white_mask_as_list(
            image, unique_person_pixel_list
        )
        persons_cutout_list = get_persons_cutout_as_list(
            image, unique_person_pixel_list
        )
    else:
        person_pixel = np.where(inst_image == 255)
        black_img = np.full(image.shape, (255, 255, 255), dtype=image.dtype)
        black_img[person_pixel] = image[person_pixel]
        persons_cutout_list = [black_img]
        persons_white_mask_list = [inst_image]
    return persons_cutout_list, persons_white_mask_list


def get_persons_white_mask_as_list(image, unique_person_pixel_list):
    persons_white_mask_list = []
    for person_pixel in unique_person_pixel_list:
        black_img = np.zeros(image.shape, dtype=image.dtype)
        black_img[person_pixel] = (255, 255, 255)
        persons_white_mask_list.append(black_img)
    return persons_white_mask_list


def get_persons_cutout_as_list(image, unique_person_pixel_list):
    persons_cutout_list = []
    for person_pixel in unique_person_pixel_list:
        black_img = np.full(image.shape, (255, 255, 255), dtype=image.dtype)
        black_img[person_pixel] = image[person_pixel]
        persons_cutout_list.append(black_img)
    return persons_cutout_list


def get_unique_person_pixel_as_list(inst_ids_img, label_ids_img):
    unique_person_pixel_list = []
    pixels_with_persons = np.where(
        (label_ids_img == PERSON_LABEL_ID) | (label_ids_img == RIDER_LABEL_ID)
    )
    pixels_with_unique_ids = inst_ids_img[pixels_with_persons]
    list_of_unique_ids = np.unique(pixels_with_unique_ids)
    for idx in list_of_unique_ids:
        unique_person_pixel = np.where(inst_ids_img == idx)
        unique_person_pixel_list.append(unique_person_pixel)
    return unique_person_pixel_list
