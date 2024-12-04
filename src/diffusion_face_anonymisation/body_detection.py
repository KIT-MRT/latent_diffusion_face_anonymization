import cv2
from ultralytics import YOLO
import logging
from pathlib import Path
from tqdm import tqdm
import json

from diffusion_face_anonymisation.body import Body


class BodyDetector:
    def __init__(self):
        self.model = YOLO("yolov8x-seg.pt")
        logging.info("YOLO model loaded successfully.")

    def body_detect_in_image(self, img_file: Path) -> list[Body]:
        logging.info(f"Starting body detection for image: {img_file}")

        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logging.info(f"Image '{img_file}' successfully read and converted to RGB.")

        results = self.model(img_rgb, retina_masks=True)
        logging.info(f"Model inference completed for image: {img_file}")

        body_list = []
        person_class_index = 0

        for result in results:
            boxes = result.boxes
            masks = result.masks
            cls = boxes.cls.tolist()

            for object_class_index, mask in zip(masks, cls):
                if object_class_index == person_class_index:
                    mask = mask.data.cpu().numpy()[0]
                    body = Body(mask)
                    body.set_body_cutout(img_rgb)
                    body_list.append(body)

        logging.info(f"Detected {len(body_list)} person(s) in image: {img_file}")
        return body_list

    def body_detect_in_files(self, image_files: list[Path], output_dir: Path):
        logging.info(f"Starting body detection for {len(image_files)} images.")
        for img_file in tqdm(image_files):
            try:
                logging.info(f"Processing image: {img_file}")
                body_list = self.body_detect_in_image(img_file)

                output_file = output_dir / f"{img_file.stem}_bodies.json"
                with open(output_file, "w+", encoding="utf8") as json_file:
                    json.dump(
                        {"masks": [body.body_mask.tolist() for body in body_list]},
                        json_file,
                    )
                logging.info(f"Results saved to: {output_file}")
            except Exception as e:
                logging.error(f"Error processing image {img_file}: {e}")
