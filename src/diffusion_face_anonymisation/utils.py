from PIL import Image
import numpy as np
import re
from pathlib import Path
import requests
from io import BytesIO
import base64

import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.body import Body

PERSON_LABEL_ID = 24
RIDER_LABEL_ID = 25


def get_image_id(full_image_string: str) -> str:
    match = re.search(r"\w+_\d+_\d+", full_image_string)
    if match:
        return match.group(0)
    else:
        raise ValueError("No match found")


def get_image_mask_dict(image_dir: str, mask_dir: str, method: str) -> dict:
    image_mask_dict = {}

    png_files = dfa_io.glob_files_by_extension(image_dir, "png")
    image_mask_dict = add_file_paths_to_image_mask_dict(png_files, image_mask_dict, "image_file")
    if method == "face":
        mask_key = "mask_file"
        json_files = dfa_io.glob_files_by_extension(mask_dir, "json")
        image_mask_dict = add_file_paths_to_image_mask_dict(json_files, image_mask_dict, mask_key)
    elif method == "body":
        mask_key = "label_ids_file"
        instance_label_files = dfa_io.glob_files_by_extension(mask_dir, "instanceIds.png")
        label_ids_files = dfa_io.glob_files_by_extension(mask_dir, "labelIds.png")
        image_mask_dict = add_file_paths_to_image_mask_dict(
            label_ids_files, image_mask_dict, mask_key
        )
        image_mask_dict = add_file_paths_to_image_mask_dict(
            instance_label_files, image_mask_dict, "instance_ids_file"
        )
    else:
        raise ValueError(f"Unknown method {method} for image mask dict creation")

    # clear image_mask_dict from entries which do not contain a mask
    image_mask_dict = {
        entry: image_mask_dict[entry]
        for entry in image_mask_dict
        if mask_key in image_mask_dict[entry]
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
        image_id = get_image_id(image_name)
        image_mask_dict.setdefault(image_id, {})[file_key] = file
    return image_mask_dict


def get_bodies_from_image(
    image: np.ndarray, unique_person_pixel_list: list[np.ndarray]
) -> list[Body]:
    bodies = []
    for person_pixel in unique_person_pixel_list:
        mask = np.zeros(image.shape, dtype=image.dtype)
        mask[person_pixel] = (255, 255, 255)
        body = Body(mask)
        body.set_body_cutout(image)
        bodies.append(body)
    return bodies


def get_unique_person_pixel_as_list(
    inst_ids_img: np.ndarray, label_ids_img: np.ndarray
) -> list[np.ndarray]:
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
    init_img_bytes = io.BytesIO()
    init_img.save(init_img_bytes, format="png")
    init_img_b64 = base64.b64encode(init_img_bytes.getvalue())

    mask_bytes = io.BytesIO()
    mask_img.save(mask_bytes, format="png")
    mask_img_b64 = base64.b64encode(mask_bytes.getvalue())
    return init_img_b64, mask_img_b64


def fill_face_payload(init_img_b64, mask_b64) -> dict:
    return {
        "init_images": ["data:image/png;base64," + init_img_b64.decode("utf-8")],
        "mask": "data:image/png;base64," + mask_b64.decode("utf-8"),
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "cfg_scale": 1,
        "sampler": "k_euler_a",
    }


def fill_body_payload(init_img_b64, mask_b64, pose_img_b64):
    payload = {
        "init_images": ["data:image/png;base64," + init_img_b64.decode("utf-8")],
        "mask": "data:image/png;base64," + mask_b64.decode("utf-8"),
        "resize_mode": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "denoising_strength": 0.65,
        "inpainting_fill": 1,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "steps": 40,
        "sampler": "k_euler_a",
        # "prompt": "RAW photo, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "negative_prompt": "nude, naked, nsfw, ugly,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        "sd_model_checkpoint": "realisticVisionV40_v40VAE-inpainting.safetensors [82e14c46c6]",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": pose_img_b64.decode("utf-8"),
                        "module": "openpose_full",
                        "model": "control_v11p_sd15_openpose [cab727d4]",
                    }
                ]
            }
        },
    }
    return payload


def send_request_to_api(png_payload: dict):
    ok = False
    for _ in range(10):
        response = requests.post(url="http://127.0.0.1:7860/sdapi/v1/img2img", json=png_payload)
        if response.status_code == 200:
            ok = True
            break

    if not ok:
        raise RuntimeError("unable to send img2img request")

    response_json = response.json()
    image_base64 = response_json["images"][0]
    return image_base64
