import numpy as np
import os
from pathlib import Path
from PIL import Image

class Body:
    def __init__(self, cutout: np.ndarray, mask: np.ndarray):
        # Store the original cutout and mask
        self.body_cutout = Image.fromarray(cutout)
        self.mask_image = Image.fromarray(mask)
        self.body_anon: Image.Image | None = None
        
    def set_anonymized_body(self, anonymized_image: np.ndarray):
        self.body_anon = Image.fromarray(anonymized_image)

    def save(self, save_path: Path, img_id: int, body_id: int):
        # Ensure the save path exists
        os.makedirs(save_path, exist_ok=True)
        # Save the cutout and mask at their original sizes
        self.body_cutout.save(f"{save_path}/body_cutout_{img_id}_{body_id}.png")
        self.mask_image.save(f"{save_path}/mask_{img_id}_{body_id}.png")
