import argparse
import json
import os

import cv2
import numpy as np

from coralml.constants import paths


def parse_csv_data_file(csv_file_path):
    """
    Read data file and return a dict containing the data
    :param csv_file_path: file containing the data
    :return: dict containing the annotation data
    """
    data = {}

    with open(csv_file_path, "r") as file:
        for line in file:
            if not line.strip():
                break

            values = line.split(" ")
            image_name = values[0] + ".JPG"

            coordinates = values[4:]
            if not len(coordinates) % 2 == 0:
                Warning("Coordinates have an uneven count for {}. skipping".format(image_name))
                continue

            annotation = {
                "class": values[2],
                "coordinates": [[int(coordinates[i]), int(coordinates[i + 1])] for i in range(0, len(coordinates), 2)]
            }

            data.setdefault(image_name, []).append(annotation)

    return data


def create_annotation_masks(data_folder_path):
    """
    Create mask images acting as annotations from the annotations data file. Mask files will be stored in the
    path.MASK_FOLDER_PATH folder

    :return: Nothing
    """

    # parse the annotations file to get the data
    data_folder_path = (data_folder_path if data_folder_path else paths.DATA_FOLDER_PATH)
    image_folder_path = os.path.join(data_folder_path, "images")
    mask_folder_path = os.path.join(data_folder_path, "masks")

    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(image_folder_path, exist_ok=True)
    os.makedirs(mask_folder_path, exist_ok=True)

    csv_file_path = os.path.join(data_folder_path, "annotations_test.csv")
    data = parse_csv_data_file(csv_file_path)

    # create a list containing all classes
    classes = list(sorted(set([annotation["class"] for img_name in data.keys() for annotation in data[img_name]])))

    # create a colour for each class, 0 is background
    colours = np.linspace(0, 255, len(classes) + 1).astype(int).tolist()

    class_mapping = {c: i + 1 for i, c in enumerate(classes)}

    colour_mapping = {"background": 0}
    colour_mapping.update({c: colours[class_mapping[c]] for c in classes})

    print("Creating masks")

    for i, image_name in enumerate(data.keys()):
        # create mask based on the size of the corresponding image
        image_path = os.path.join(image_folder_path, image_name)
        image_height, image_width = cv2.imread(image_path).shape[:2]
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # go through each annotation entry and fill the corresponding polygon. Color corresponds to class
        for annotation in data[image_name]:
            colour = colour_mapping[annotation["class"]]

            points = annotation["coordinates"]
            cv2.fillPoly(mask, [np.array(points)], color=colour)

        # save the mask
        name, _ = os.path.splitext(image_name)
        out_name = name + "_mask.png"
        print(f"Saving {os.path.join(mask_folder_path, out_name)}")
        cv2.imwrite(os.path.join(mask_folder_path, out_name), mask)

    # write color mapping to file
    with open(os.path.join(data_folder_path, "colour_mapping.json"), "w") as fp:
        json.dump(colour_mapping, fp, indent=4)


def correct_masks():
    """
    There are some masks that need to be rotated by 180 degree to fit the data. apparently there has been a problem
    during annotation or so. This method is only useful for the current version (pre 1.4) of the data. The data on the
    server should soon be fixed.
    :return:
    """
    mask_files = ["2018_0729_112409_029_mask.png",
                  "2018_0729_112442_045_mask.png",
                  "2018_0729_112537_066_mask.png",
                  "2018_0729_112536_063_mask.png",
                  "2018_0729_112541_053_mask.png",
                  "2018_0729_112455_038_mask.png",
                  "2018_0729_112449_036_mask.png"]

    for mask_file in mask_files:
        path = os.path.join(paths.MASK_FOLDER_PATH, mask_file)
        mask = cv2.imread(path)
        mask = mask[::-1, ::-1, 0]
        cv2.imwrite(path, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder_path', default=None, type=str,
                        help='Path to the data directory, where to save the outputs.')
    data_folder_path = parser.parse_args().data_folder_path
    create_annotation_masks(data_folder_path=data_folder_path)
