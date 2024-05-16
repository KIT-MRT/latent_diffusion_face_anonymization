import os
from pathlib import Path
import yaml
import argparse


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


def glob_files_by_extension(base_dir: str, extension: str) -> list:
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
