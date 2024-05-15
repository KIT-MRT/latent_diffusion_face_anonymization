import json
from PIL import Image
import numpy as np
import requests
import io
from io import BytesIO
import base64
from skimage.filters import gaussian


class FaceBoundingBox:
    def __init__(self, bounding_box_list: list):
        self.xtl = bounding_box_list[0]
        self.ytl = bounding_box_list[3]
        self.xbr = bounding_box_list[2]
        self.ybr = bounding_box_list[1]
        self.confidence = bounding_box_list[4]

    def get_slice_area(self) -> tuple[slice, slice]:
        return (slice(self.ytl, self.ybr), slice(self.xtl, self.xbr))


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
        try:
            image_mask_dict[image_name][file_key] = file
        except KeyError:
            image_mask_dict[image_name] = {}
            image_mask_dict[image_name][file_key] = file
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


def encode_image_mask_to_b64(init_img, mask_img):
    init_img_bytes = BytesIO()
    init_img.save(init_img_bytes, format="png")
    init_img_b64 = base64.b64encode(init_img_bytes.getvalue())

    mask_bytes = BytesIO()
    mask_img.save(mask_bytes, format="png")
    mask_img_b64 = base64.b64encode(mask_bytes.getvalue())
    return init_img_b64, mask_img_b64


def fill_png_payload(init_img_b64, mask_b64):
    return {
        "init_images": ["data:image/png;base64," + init_img_b64.decode("utf-8")],
        "mask": "data:image/png;base64," + mask_b64.decode("utf-8"),
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "cfg_scale": 1,
        "sampler": "k_euler_a",
    }


def send_request_to_api(png_payload):
    ok = False
    for _ in range(10):
        response = requests.post(
            url=f"http://127.0.0.1:7860/sdapi/v1/img2img", json=png_payload
        )
        if response.status_code == 200:
            ok = True
            break

    if not ok:
        raise RuntimeError("unable to send img2img request")

    response_json = response.json()
    image_base64 = response_json["images"][0]
    return image_base64


def request_inpaint(init_img, mask):
    init_img_b64, mask_b64 = encode_image_mask_to_b64(init_img, mask)
    png_payload = fill_png_payload(init_img_b64, mask_b64)
    inpainted_img_b64 = send_request_to_api(png_payload)

    def convert_b64_to_pil(img_b64):
        return Image.open(io.BytesIO(base64.b64decode(img_b64.split(",", 1)[0])))

    return convert_b64_to_pil(inpainted_img_b64)


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
