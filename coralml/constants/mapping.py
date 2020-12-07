import json
import os

from coralml.constants.paths import DATA_FOLDER_PATH


def get_colour_mapping(data_folder_path=None):
    data_folder_path = data_folder_path or DATA_FOLDER_PATH

    file_path = os.path.join(data_folder_path, "colour_mapping.json")
    with open(file_path, "r") as fp:
        colour_mapping = json.load(fp)
    return colour_mapping
