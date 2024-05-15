import os
from pathlib import Path
import yaml


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
