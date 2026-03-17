import numpy as np
import os
from pathlib import Path
from PIL import Image


class LicensePlate:
    """Represents a detected license plate with oriented bounding box and anonymization state."""

    def __init__(self, oriented_bbox: np.ndarray):
        """
        Initialize a LicensePlate object.
        
        Args:
            oriented_bbox: 4 corner points from YOLO11-OBB as numpy array or list
                          Shape: (4, 2) or flattened to 8 values
                          [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Store original oriented bounding box
        self.oriented_bbox = np.array(oriented_bbox, dtype=np.float32)
        if self.oriented_bbox.shape == (8,):
            self.oriented_bbox = self.oriented_bbox.reshape(4, 2)
        
        # Compute axis-aligned bounding box from oriented corners
        self.aabb = self._get_axis_aligned_bbox(self.oriented_bbox)
        
        # Cutout and anonymized images
        self.lp_cutout: Image.Image | None = None
        self.lp_anon: Image.Image | None = None

    def _get_axis_aligned_bbox(self, obb: np.ndarray) -> tuple[int, int, int, int]:
        """
        Convert oriented bounding box corners to axis-aligned bounding box.
        
        Args:
            obb: 4 corner points shape (4, 2)
            
        Returns:
            Tuple of (y_min, y_max, x_min, x_max) for numpy array slicing
        """
        x_min = int(np.floor(obb[:, 0].min()))
        x_max = int(np.ceil(obb[:, 0].max()))
        y_min = int(np.floor(obb[:, 1].min()))
        y_max = int(np.ceil(obb[:, 1].max()))
        
        return (y_min, y_max, x_min, x_max)

    def set_lp_cutout(self, image: np.ndarray):
        """
        Extract license plate region from image using axis-aligned bounding box.
        
        Args:
            image: Input image as numpy array (H, W, C)
        """
        y_min, y_max, x_min, x_max = self.aabb
        
        # Clamp to image bounds
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)
        
        cutout = image[y_min:y_max, x_min:x_max]
        self.lp_cutout = Image.fromarray(cutout.astype(np.uint8))

    def add_anon_lp_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Composite anonymized license plate back into original image.
        
        Args:
            image: Original image as numpy array
            
        Returns:
            Image with anonymized LP composited back
        """
        if self.lp_anon is None:
            return image
        
        y_min, y_max, x_min, x_max = self.aabb
        
        # Clamp to image bounds
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)
        
        # Handle case where anonymized region might have different dimensions
        anon_array = np.array(self.lp_anon)
        
        # Resize if necessary (shouldn't happen if cutout logic is correct)
        region_h = y_max - y_min
        region_w = x_max - x_min
        if anon_array.shape[0] != region_h or anon_array.shape[1] != region_w:
            anon_img = Image.fromarray(anon_array)
            anon_img = anon_img.resize((region_w, region_h), Image.Resampling.LANCZOS)
            anon_array = np.array(anon_img)
        
        image[y_min:y_max, x_min:x_max] = anon_array
        return image

    def save(self, save_path: Path, img_id: int, lp_id: int):
        """
        Save anonymized license plate for debugging.
        
        Args:
            save_path: Directory to save in
            img_id: Image ID for naming
            lp_id: License plate ID for naming
        """
        os.makedirs(save_path, exist_ok=True)
        if self.lp_anon:
            self.lp_anon.save(f"{save_path}/lp_anon_{img_id}_{lp_id}.png")

    def __repr__(self):
        return (
            f"LicensePlate(bbox={self.aabb}, "
            f"cutout_size={self.lp_cutout.size if self.lp_cutout else None}, "
            f"anon={self.lp_anon is not None})"
        )
