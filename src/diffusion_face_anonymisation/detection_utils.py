import logging
import os
from tqdm import tqdm
from pathlib import Path

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
