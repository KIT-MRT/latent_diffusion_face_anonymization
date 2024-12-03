import logging
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json

from retinaface.RetinaFace import detect_faces as retina_face_detect_faces


def detect_face_in_image(*, path_to_image_file: Path, threshold=0.3):
    logging.info(
        f"Starting face detection for image: {path_to_image_file} with threshold: {threshold}"
    )

    faces_detected = retina_face_detect_faces(
        img_path=str(path_to_image_file), threshold=threshold
    )
    if not faces_detected:
        logging.warning(f"No faces detected in image: {path_to_image_file}")
        return []

    faces = []
    for k in faces_detected:
        if not isinstance(k, str):
            logging.error(
                f"Invalid key type detected: {k} in image: {path_to_image_file}"
            )
            continue

        face = faces_detected[k]
        bl = [face["facial_area"][0], face["facial_area"][1]]
        tr = [face["facial_area"][2], face["facial_area"][3]]
        h = tr[1] - bl[1]

        conf = face["score"]
        tl = [bl[0], bl[1] + h]
        br = [tr[0], tr[1] - h]
        faces.append([int(tl[0]), int(tl[1]), int(br[0]), int(br[1]), float(conf)])

    logging.info(f"Detected {len(faces)} face(s) in image: {path_to_image_file}")
    return faces


def detect_faces_in_files(image_files: list[Path], image_dir: Path, output_dir: Path):
    for path_to_image_file in tqdm(image_files):
        logging.info(f"Processing image: {path_to_image_file}")
        faces = detect_face_in_image(path_to_image_file=path_to_image_file)

        output_file = f"{output_dir}/{os.path.splitext(os.path.basename(path_to_image_file))[0]}.json"
        logging.info(f"Output written to file: {output_file}")
        with open(output_file, "w+", encoding="utf8") as json_file:
            json.dump({"face": faces}, json_file)


class BodyDetector:
    def __init__(self):
        self.model = YOLO("yolov8x-seg.pt")
        logging.info("YOLO model loaded successfully.")

    def body_detect_in_image(self, img_file: Path):
        logging.info(f"Starting body detection for image: {img_file}")

        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logging.info(f"Image '{img_file}' successfully read and converted to RGB.")

        results = self.model(img_rgb, retina_masks=True)
        logging.info(f"Model inference completed for image: {img_file}")

        persons_cutout = []
        persons_white_mask = []
        person_class_index = 0

        for result in results:
            boxes = result.boxes
            masks = result.masks
            cls = boxes.cls.tolist()

            for i in range(len(masks)):
                if int(cls[i]) == person_class_index:
                    mask = masks[i].data.cpu().numpy()[0]
                    person_pixel = np.where(mask == 1)

                    cutout_img = np.full(
                        img_rgb.shape, (255, 255, 255), dtype=img_rgb.dtype
                    )
                    cutout_img[person_pixel] = img_rgb[person_pixel]

                    persons_cutout.append(cutout_img)
                    persons_white_mask.append(mask)

        logging.info(f"Detected {len(persons_cutout)} person(s) in image: {img_file}")
        return persons_cutout, persons_white_mask

    def body_detect_in_files(self, image_files: list[Path], output_dir: Path):
        logging.info(f"Starting body detection for {len(image_files)} images.")
        for img_file in tqdm(image_files):
            try:
                logging.info(f"Processing image: {img_file}")
                _, persons_white_mask = self.body_detect_in_image(img_file)

                output_file = output_dir / f"{img_file.stem}_bodies.json"
                with open(output_file, "w+", encoding="utf8") as json_file:
                    json.dump(
                        {"masks": [mask.tolist() for mask in persons_white_mask]},
                        json_file,
                    )
                logging.info(f"Results saved to: {output_file}")
            except Exception as e:
                logging.error(f"Error processing image {img_file}: {e}")
