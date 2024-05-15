import argparse
import os
from tqdm import tqdm
from retinaface import RetinaFace
import logging
import json


def detect(*, img_path, threshold=0.3):
    logger.info(
        f"Starting face detection for image: {img_path} with threshold: {threshold}"
    )

    faces_detected = RetinaFace.detect_faces(img_path=img_path, threshold=threshold)
    if not faces_detected:
        logger.warning(f"No faces detected in image: {img_path}")
        return []

    faces = []
    for k in faces_detected:
        if not isinstance(k, str):
            logger.error(f"Invalid key type detected: {k} in image: {img_path}")
            continue

        face = faces_detected[k]
        bl = [face["facial_area"][0], face["facial_area"][1]]
        tr = [face["facial_area"][2], face["facial_area"][3]]
        h = tr[1] - bl[1]

        conf = face["score"]
        tl = [bl[0], bl[1] + h]
        br = [tr[0], tr[1] - h]
        faces.append([int(tl[0]), int(tl[1]), int(br[0]), int(br[1]), float(conf)])

    logger.info(f"Detected {len(faces)} face(s) in image: {img_path}")
    return faces


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("face_detection.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", required=True, type=str, help="Path to input directory"
    )
    parser.add_argument(
        "--mask_dir", required=True, type=str, help="Path to masks (output) directory"
    )
    args = parser.parse_args()

    input_path = args.image_dir
    output_path = args.mask_dir

    if not os.path.exists():
        logger.error(f"Input directory does not exist: {input_path}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Output directory created: {output_path}")

    img_paths = list(os.listdir(input_path))
    for image in tqdm(img_paths):
        img_path = f"{input_path}/{image}"
        logger.info(f"Processing image: {img_path}")
        faces = detect(img_path=img_path)

        os.makedirs(f"{output_path}", exist_ok=True)
        output_file = (
            f"{output_path}/{os.path.splitext(os.path.basename(img_path))[0]}.json"
        )
        logger.info(f"Output written to file: {output_file}")
        with open(output_file, "w+", encoding="utf8") as json_file:
            json.dump({"face": faces}, json_file)
