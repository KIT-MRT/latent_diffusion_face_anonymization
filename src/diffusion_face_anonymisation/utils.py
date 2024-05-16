import json
from PIL import Image
import numpy as np

import diffusion_face_anonymisation.io_functions as dfa_io


class FaceBoundingBox:
    def __init__(self, bounding_box_list: list):
        self.xtl = bounding_box_list[0]
        self.ytl = bounding_box_list[3]
        self.xbr = bounding_box_list[2]
        self.ybr = bounding_box_list[1]
        self.confidence = bounding_box_list[4]

    def get_slice_area(self) -> tuple[slice, slice]:
        return (slice(self.ytl, self.ybr), slice(self.xtl, self.xbr))


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


def get_face_bounding_box_list_from_file(path_to_bounding_box_file: str) -> dict:
    with open(path_to_bounding_box_file, "r") as bounding_box_file_json:
        bb_dict = json.load(bounding_box_file_json)
    return bb_dict["face"]


def convert_bb_to_mask_dict_list(
    all_faces_list: list, image_width: int, image_height: int
) -> list:
    mask_dict_list = list()
    for face_bb_list in all_faces_list:
        mask_image_np = np.zeros((image_height, image_width), np.uint8)
        face_bb = FaceBoundingBox(face_bb_list)
        mask_image_np[face_bb.get_slice_area()] = 255
        mask_dict_list.append({"bb": face_bb, "mask": mask_image_np})
    return mask_dict_list


def add_file_paths_to_image_mask_dict(
    file_paths: list, image_mask_dict: dict, file_key: str
) -> dict:
    for file in file_paths:
        image_name = file.stem
        image_mask_dict.setdefault(image_name, {})[file_key] = file
    return image_mask_dict


def add_inpainted_faces_to_orig_img(
    image: np.ndarray, inpainted_img_list: list, mask_dict_list: list
):
    img_np = np.array(image)
    for inpainted_img, mask_dict in zip(inpainted_img_list, mask_dict_list):
        face_bb = mask_dict["bb"]
        face_slice_area = face_bb.get_slice_area()
        inpainted_img_np = np.array(inpainted_img)
        img_np[face_slice_area] = inpainted_img_np[face_slice_area]
    return Image.fromarray(img_np)


def get_face_cutout(image: np.ndarray, mask_dict: dict):
    face_slice = mask_dict["bb"].get_slice_area()
    face_cutout_np = image[face_slice]
    return Image.fromarray(face_cutout_np)
