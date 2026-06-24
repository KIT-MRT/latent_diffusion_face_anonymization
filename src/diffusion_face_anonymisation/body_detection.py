import cv2
import logging
from pathlib import Path
import numpy as np
from typing import List

from diffusion_face_anonymisation.body import Body


class BodyDetector:
    def __init__(self, batch_size: int = 8):
        from ultralytics import YOLO
        self.model = YOLO("yolo12l-person-seg-extended.pt", verbose=False)
        self.batch_size = batch_size
        logging.info(f"YOLO model loaded successfully (batch_size={batch_size}).")

    def body_detect_in_image(self, img_file: Path) -> list[Body]:
        logging.info(f"Starting body detection for image: {img_file}")

        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logging.info(f"Image '{img_file}' successfully read and converted to RGB.")

        results = self.model(img_rgb, retina_masks=True, verbose=False)
        logging.info(f"Model inference completed for image: {img_file}")

        body_list = self._parse_results(results, img_rgb)

        logging.info(f"Detected {len(body_list)} person(s) in image: {img_file}")
        return body_list

    def batch_detect(self, img_files: List[Path]) -> dict[Path, list[Body]]:
        """Detect bodies in a batch of images.

        This is much faster than processing images one by one as it
        maximizes GPU utilization.

        Args:
            img_files: List of image file paths

        Returns:
            Dictionary mapping image file path to list of detected Body objects
        """
        logging.info(f"Starting batch body detection for {len(img_files)} images")

        # Load all images
        images = []
        valid_files = []
        for img_file in img_files:
            try:
                img_bgr = cv2.imread(str(img_file))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    valid_files.append(img_file)
            except Exception as e:
                logging.error(f"Error loading image {img_file}: {e}")

        if not images:
            return {}

        # Batch inference - single GPU call for all images
        results = self.model(images, retina_masks=True)

        # Parse results
        bodies_dict = {}
        for img_file, result, img_rgb in zip(valid_files, results, images):
            body_list = self._parse_result(result, img_rgb)
            bodies_dict[img_file] = body_list
            logging.info(f"Detected {len(body_list)} person(s) in image: {img_file}")

        logging.info(f"Batch detection complete for {len(valid_files)} images")
        return bodies_dict

    def _parse_result(self, result, img_rgb: np.ndarray) -> list[Body]:
        """Parse a single YOLO result into Body objects"""
        body_list = []
        masks = result.masks

        if not masks:
            return body_list

        for mask in masks:
            mask = mask.data.cpu().numpy().astype(np.uint8)[0]
            if mask.sum() > 10:
                mask_img = np.zeros_like(img_rgb)
                person_pixel = np.where(mask == 1)
                mask_img[person_pixel] = (255, 255, 255)
                body = Body(mask_img)
                body.set_body_cutout(img_rgb)
                body_list.append(body)

        return body_list

    def _parse_results(self, results, img_rgb: np.ndarray) -> list[Body]:
        """Parse YOLO results (single image) into Body objects"""
        body_list = []

        for result in results:
            masks = result.masks
            if not masks:
                continue

            for mask in masks:
                mask = mask.data.cpu().numpy().astype(np.uint8)[0]
                if mask.sum() > 10:
                    mask_img = np.zeros_like(img_rgb)
                    person_pixel = np.where(mask == 1)
                    mask_img[person_pixel] = (255, 255, 255)
                    body = Body(mask_img)
                    body.set_body_cutout(img_rgb)
                    body_list.append(body)

        return body_list

    def body_detect_in_files(self, img_file: Path) -> list[Body]:
        body_list = []
        try:
            logging.info(f"Processing image: {img_file}")
            body_list = self.body_detect_in_image(img_file)
        except Exception as e:
            logging.error(f"Error processing image {img_file}: {e}")
        return body_list
