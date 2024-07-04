import os
from pathlib import Path
import yaml

def glob_files_by_extension(base_dir, extension):
    files_list = []
    for root, _, files in os.walk(base_dir):
        files_list.extend(
            [Path(root, file) for file in files if file.endswith(extension)]
        )
    return files_list

def load_config(config_yaml):
    with open(config_yaml, "r") as config_yaml_file:
        config_dict = yaml.load(config_yaml_file, yaml.SafeLoader)
    return config_dict


def load_images(img_path):
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return {"path": img_path, "img": image}
