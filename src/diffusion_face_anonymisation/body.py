import os
from pathlib import Path
import numpy as np
from PIL import Image


class Body:
    def __init__(self, mask: np.ndarray):
        self.body_cutout: Image.Image
        assert mask.dtype == np.uint8
        self.body_mask = mask
        self.body_mask_image: Image.Image
        self.body_cutout_resized: Image.Image
        self.body_mask_resized: Image.Image
        self.body_anon: Image.Image | None = None

    def set_body_cutout(self, image: np.ndarray):
        body_cutout = np.zeros((image.shape), dtype=np.uint8)
        idx = np.where(self.body_mask > 0)
        body_cutout[idx] = image[idx]
        self.body_cutout = Image.fromarray(body_cutout)

    def resize(self, width: int, height: int):
        self.body_cutout_resized = self.body_cutout.resize((width, height))

    def add_anon_body_to_image(self, image: np.ndarray) -> np.ndarray:
        body_anon_np = np.array(self.body_anon)
        body_mask_np = np.array(self.body_mask)
        image[body_mask_np == 255] = body_anon_np[body_mask_np == 255]
        return image

    def save(self, save_path: Path, img_id: int, body_id: int):
        os.makedirs(save_path, exist_ok=True)
        self.body_cutout.save(f"{save_path}/body_cutout_{img_id}_{body_id}.png")
        if self.body_anon:
            self.body_anon.save(f"{save_path}/body_anon_{img_id}_{body_id}.png")


def add_body_cutout_and_mask_img(bodies: list[Body], image: np.ndarray) -> list[Body]:
    for body in bodies:
        body.set_body_cutout(image)
        mask_image_np = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        body.body_mask_image = Image.fromarray(mask_image_np)
    return bodies
