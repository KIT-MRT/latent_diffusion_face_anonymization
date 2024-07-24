import cv2
import numpy as np
from ultralytics import YOLO
import os


# TODO: implement this class once with yolov8 and once with semantic SAM
class BodyDetector:
    def __init__(self):
        pass

    #We Initiate by taking Base_path of Images from body_mask_detection_test and calling the model
    def __init__(self, test_image_base_path):
        self.test_image_base_path = test_image_base_path
        self.model = YOLO("yolov8n-seg.pt")
    #Reading the images
    def detect(self, img_file):
        #TODO: do the actual detection call of the corresponding network and return the persons as cutouts and white masks
        persons_cutout = None
        persons_white_mask = None
        img_path = os.path.join(self.test_image_base_path, img_file)


        #Making sure we have readable image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file '{img_path}' not found.")

        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape

        if img is None:
            raise RuntimeError(f"Failed to read image file '{img_path}'.")
        #Run YOLO on image
        results = self.model(img_path) 
        #Storing Cutout and White Mask as List
        persons_cutout = []
        persons_white_mask = []
        person_class_index = 0



        for result in results:
            boxes = result.boxes
            masks = result.masks
            cls = boxes.cls.tolist()


            for i in range(len(masks)):
                #If detected class is person we obtain mask and convert it to numpy array
                if int(cls[i]) == person_class_index:
                    mask = masks[i].data.cpu().numpy()[0]
                    #Getting a resized mask with same dimension of original image using nearest-neighbor interpolation for accurate overlay of mask
                    mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours((mask_resized * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    #Process contours to create cutout
                    for contour in contours:
                        # Create a blank image to draw contours
                        cutout_img = np.zeros_like(img)
                        cv2.drawContours(cutout_img, [contour], -1, (0, 255, 0), 2)

                        persons_cutout.append(cutout_img)
                        persons_white_mask.append(mask_resized)

        return persons_cutout, persons_white_mask
