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
        
        # Handle grayscale masks (from SAM3) vs RGB masks (from YOLO)
        if len(body_mask_np.shape) == 3:
            body_mask_np = body_mask_np[:, :, 0]  # Convert RGB to grayscale
        
        # Ensure binary mask
        body_mask_np = (body_mask_np > 127).astype(np.uint8)
        
        # Apply anonymization - mask indices need to match
        if body_anon_np.shape[:2] != body_mask_np.shape:
            # Resize anon to match mask if needed
            h, w = body_mask_np.shape
            body_anon_np = np.array(Image.fromarray(body_anon_np).resize((w, h)))
        
        image[body_mask_np == 1] = body_anon_np[body_mask_np == 1]
        return image

    def save(self, save_path: Path, img_id: int, body_id: int):
        os.makedirs(save_path, exist_ok=True)
        self.body_cutout.save(f"{save_path}/body_cutout_{img_id}_{body_id}.png")
        self.body_mask_image.save(f"{save_path}/mask_{img_id}_{body_id}.png")
        if self.body_anon:
            self.body_anon.save(f"{save_path}/body_anon_{img_id}_{body_id}.png")

    def __str__(self):
        return_string = "Body Object\n"
        return_string += f" - Body cutout size: {self.body_cutout.size}\n"
        return_string += f" - Body mask size: {self.body_mask_image.size}\n"
        return_string += f" - Body mask content sum: {np.sum(self.body_mask)}\n"
        return return_string


def add_body_cutout_and_mask_img(bodies: list[Body], image: np.ndarray) -> list[Body]:
    for body in bodies:
        body.set_body_cutout(image)
        body.body_mask_image = Image.fromarray(body.body_mask)
    return bodies
