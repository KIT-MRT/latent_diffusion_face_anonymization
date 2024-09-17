import os
from pathlib import Path
import yaml
import argparse
import logging
import json
from diffusion_face_anonymisation.face import Face


def setup_parser_and_parse_args() -> tuple[Path, Path, Path, str]:
    parser = argparse.ArgumentParser(
        prog="Naive Face Anonymization",
        description="Anonymize Faces with naive functions.",
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument(
        "--anon_function",
        type=str,
        required=True,
        choices=["white", "gauss", "pixel", "ldfa"],
    )
    args = parser.parse_args()
    return (
        Path(args.image_dir),
        Path(args.mask_dir),
        Path(args.output_dir),
        args.anon_function,
    )


def setup_face_detection_parser_and_parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(
        prog="Face detection",
        description="Detects faces with Retina face and writes the bounding box coordinates to a json",
    )
    parser.add_argument(
        "--image_dir", required=True, type=str, help="Path to input directory"
    )
    parser.add_argument(
        "--mask_dir", required=True, type=str, help="Path to masks (output) directory"
    )
    args = parser.parse_args()

    return args.image_dir, args.mask_dir


def glob_files_by_extension(base_dir: str, extension: str) -> list[Path]:
    files_list = []
    for root, _, files in os.walk(base_dir):
        files_list.extend(
            [Path(root, file) for file in files if file.endswith(extension)]
        )
    return files_list


def load_config(config_yaml: str) -> dict:
    with open(config_yaml, "r") as config_yaml_file:
        config_dict = yaml.load(config_yaml_file, yaml.SafeLoader)
    return config_dict


def get_faces_from_file(face_mask_file_path: Path) -> list[Face]:
    with open(face_mask_file_path, "r") as face_mask_file:
        faces_dict = json.load(face_mask_file)
    faces = []
    for face in faces_dict["face"]:
        faces.append(Face(face))
    return faces


def save_anon_image(anon_img, image_file: str, output_dir: Path, anon_function: str):
    # TODO: implement debug img saving
    orig_file = Path(image_file)
    # Construct paths for output and debug images
    output_filename = f"{orig_file.stem}_anon_{anon_function}{orig_file.suffix}"

    # Save the final image and debug image
    anon_img.save(output_dir / output_filename)

    logging.info(f"Anonymized image saved to {output_dir/output_filename}")
