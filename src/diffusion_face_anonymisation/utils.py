from PIL import Image
import numpy as np
from pathlib import Path
import requests
from io import BytesIO
import base64

import diffusion_face_anonymisation.io_functions as dfa_io


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


def encode_image_mask_to_b64(init_img: Image.Image, mask_img: Image.Image) -> tuple[bytes, bytes]:
    init_img_bytes = BytesIO()
    init_img.save(init_img_bytes, format="png")
    init_img_b64 = base64.b64encode(init_img_bytes.getvalue())

    mask_bytes = BytesIO()
    mask_img.save(mask_bytes, format="png")
    mask_img_b64 = base64.b64encode(mask_bytes.getvalue())
    return init_img_b64, mask_img_b64


def fill_png_payload(init_img_b64, mask_b64) -> dict:
    return {
        "init_images": ["data:image/png;base64," + init_img_b64.decode("utf-8")],
        "mask": "data:image/png;base64," + mask_b64.decode("utf-8"),
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "cfg_scale": 1,
        "sampler": "k_euler_a",
    }


def send_request_to_api(png_payload: dict):
    ok = False
    for _ in range(10):
        response = requests.post(
            url="http://127.0.0.1:7860/sdapi/v1/img2img", json=png_payload
        )
        if response.status_code == 200:
            ok = True
            break

    if not ok:
        raise RuntimeError("unable to send img2img request")

    response_json = response.json()
    image_base64 = response_json["images"][0]
    return image_base64
