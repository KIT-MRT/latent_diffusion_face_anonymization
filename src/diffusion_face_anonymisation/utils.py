import json
from PIL import Image
import numpy as np
from pathlib import Path

import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.bounding_box_utils import Face


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


def get_faces_from_file(path_to_bounding_box_file: Path) -> list[Face]:
    with open(path_to_bounding_box_file, "r") as bounding_box_file_json:
        bb_dict = json.load(bounding_box_file_json)
    faces = []
    for face in bb_dict["face"]:
        faces.append(Face(face))
    return faces


def add_face_cutout_and_mask_img(faces: list[Face], image: np.ndarray):
    for face in faces:
        face.set_face_cutout(image)
        mask_image_np = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        mask_image_np[face.bounding_box.get_slice_area()] = 255
        face.set_mask_image(mask_image_np)


def add_file_paths_to_image_mask_dict(
    file_paths: list[Path], image_mask_dict: dict, file_key: str
) -> dict:
    for file in file_paths:
        image_name = file.stem
        image_mask_dict.setdefault(image_name, {})[file_key] = file
    return image_mask_dict


def add_inpainted_faces_to_orig_img(
    image: np.ndarray, inpainted_img_list: list[Image.Image], mask_dict_list: list[dict]
) -> Image.Image:
    img_np = np.array(image)
    for inpainted_img, mask_dict in zip(inpainted_img_list, mask_dict_list):
        face_bb = mask_dict["bounding_box"]
        face_slice_area = face_bb.get_slice_area()
        inpainted_img_np = np.array(inpainted_img)
        img_np[face_slice_area] = inpainted_img_np[face_slice_area]
    return Image.fromarray(img_np)
