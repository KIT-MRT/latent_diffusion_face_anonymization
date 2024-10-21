import numpy as np
import os
from pathlib import Path
from PIL import Image

class Body:
    def __init__(self, mask: np.ndarray):
        
        self.body_cutout: Image.Image   
        self.body_mask = mask
        self.body_mask_image: Image.Image = Image.fromarray(mask.astype(np.uint8))
        self.body_cutout_resized: Image.Image 
        self.body_mask_resized: Image.Image 
        self.body_anon: Image.Image | None = None  

    def set_body_cutout(self, image: np.ndarray):
        mask_3d = np.stack([np.array(self.body_mask_image)] * 3, axis=-1)  
        self.body_cutout = Image.fromarray(np.where(mask_3d == 1, image, 0).astype(np.uint8))
    
    def resize(self, width: int, height: int):
        self.body_cutout_resized = self.body_cutout.resize((width, height))
        self.body_mask_resized = self.body_mask_image.resize((width, height))

    def add_anon_body_to_image(self, image: np.ndarray) -> np.ndarray:
        body_anon_np = np.array(self.body_anon)
        body_mask_np = np.array(self.body_mask)
        image[body_mask_np > 0] = body_anon_np[body_mask_np > 0]
        return image


def add_body_cutout_and_mask_img(bodies: list[Body], image: np.ndarray):
    for body in bodies:
        body.set_body_cutout(image)  
        mask_image_np = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        mask_image_np[body.body_mask > 0] = 255  
        body.body_mask = Image.fromarray(mask_image_np)  
    return bodies
