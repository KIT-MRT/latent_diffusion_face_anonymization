import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch

class BodyDetector:
    def __init__(self):
        self.model = YOLO("yolov8x-seg.pt")
    
    def detect(self, img_file):
        # Ensure the image file exists
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image file '{img_file}' not found.")      
        # Read the image
        img_bgr = cv2.imread(img_file)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image file '{img_file}'.")
        
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img_rgb.shape
        
        # Run YOLO on the RGB image
        results = self.model(img_rgb,retina_masks=True)
        
        # Initialize lists to store cutouts and masks
        persons_cutout = []
        persons_white_mask = []
        person_class_index = 0  # Assuming 'person' is class index 0
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            cls = boxes.cls.tolist()
        
            for i in range(len(masks)):
                # If detected class is person, obtain mask and convert it to numpy array
                if int(cls[i]) == person_class_index:
                    mask = masks[i].data.cpu().numpy()[0]
                    # Find pixels that are part of the person
                    person_pixel = np.where(mask == 1)

                    # Create a white image
                    cutout_img = np.full(img_rgb.shape, (255, 255, 255), dtype=img_rgb.dtype)

                    # Cut out the person from the original image
                    cutout_img[person_pixel] = img_rgb[person_pixel]

                    # Convert the cutout image back to BGR for display with OpenCV if needed
                    #cutout_img_bgr = cv2.cvtColor(cutout_img, cv2.COLOR_RGB2BGR)

                    # Append the cutout image and mask to the respective lists
                    persons_cutout.append(cutout_img)
                    persons_white_mask.append(mask)
        
        return persons_cutout, persons_white_mask

