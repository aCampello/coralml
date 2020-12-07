import argparse
from collections import defaultdict
import os
import json
import sys

import cv2
import torch
from tqdm import tqdm
import numpy as np

from coralml.constants import paths, mapping
from coralml.ml import predict
from coralml.ml.utils import colour_mask_to_class_id_mask, cut_windows

sys.path.extend([paths.DEEPLAB_FOLDER_PATH, os.path.join(paths.DEEPLAB_FOLDER_PATH, "utils")])

from metrics import Evaluator

from coralml.ml.utils import load_model, colour_mask_to_class_id_mask


def _calculate_iou(intersection_per_substrate, union_per_substrate, classes_map):
    eps = 10e-16
    # Ads epsilon to the denominator to avoid division by zero
    iou_per_substrate = {
        substrate_name: intersection_per_substrate[substrate_name] / (union_per_substrate[substrate_name] + eps)
        for substrate_name in classes_map.values()
    }

    iou_average = sum(intersection_per_substrate.values()) / \
                  sum(union_per_substrate.values())

    return iou_per_substrate, iou_average


def evaluate(image_file_paths, gt_file_paths, model_path,
             nn_input_size, window_sizes=None,
             step_sizes=None,
             device=None,
             data_folder_path=None,
             log_file_path=None):
    """
    Evaluate model performance
    :param image_file_paths: paths to images that will be predicted
    :param gt_file_paths: paths to ground truth masks
    :param model: model used for prediction
    :param nn_input_size: size that the images will be scaled to before feeding them into the nn
    :param num_classes: number of classes
    :param window_sizes: list of sizes that determine how the image will be cut (for sliding window). Image will be cut
    into squares
    :param step_sizes: list of step sizes for sliding window
    :param device: PyTorch device (cpu or gpu)
    :return: list of prediction masks if res_fcn is None, else Nothing
    """

    model = load_model(model_path)
    colour_mapping = mapping.get_colour_mapping(data_folder_path=data_folder_path)
    classes_map = {x: y for x, y in enumerate(sorted(colour_mapping.keys()))}

    with torch.no_grad():
        predictions = predict.predict(image_file_paths=image_file_paths,
                                      model=model,
                                      nn_input_size=nn_input_size,
                                      window_sizes=window_sizes,
                                      step_sizes=step_sizes,
                                      device=device)

    # Calculates IoU looping over images and masks

    intersection_per_substrate = defaultdict(int)
    union_per_substrate = defaultdict(int)

    for gt_file_path, prediction in zip(gt_file_paths, predictions):
        pred_class_id_mask = np.argmax(prediction, axis=-1)
        gt_colour_mask = cv2.imread(gt_file_path)[:, :, 0]
        gt_class_id_mask = colour_mask_to_class_id_mask(gt_colour_mask)

        for substrate_idx, substrate_name in classes_map.items():
            intersection, union = calculate_agreement(gt_class_id_mask, pred_class_id_mask,
                                                      substrate_idx=substrate_idx)
            if union:
                intersection_per_substrate[substrate_name] += intersection
                union_per_substrate[substrate_name] += union

        iou_per_substrate, iou_average = _calculate_iou(
            intersection_per_substrate,
            union_per_substrate,
            classes_map
        )

        # Gradually append accuracy stats

        if log_file_path:
            with open(log_file_path, 'w') as f:
                f.write(json.dumps(iou_per_substrate, indent=4))
                f.write('\n')
                f.write(f"avg: {iou_average}")

    return iou_per_substrate, iou_average


def calculate_agreement(pixel_image_1, pixel_image_2, substrate_idx=None):
    """
    Given two images containing pixel annotations, calculates agreement.
    If a substrate_idx is provided, calculates IoU over that substrate,
    otherwise calculates pixel accuracy

    Args:
        pixel_image_1: a numpy array
        pixel_image_2: a numpy array
        substrate_idx: a valid substrate index
    Returns:
        Agreement metric (accuracy or IoU)
    """

    intersection = ((pixel_image_1 == substrate_idx) *
                    (pixel_image_2 == substrate_idx)).sum()

    union = ((pixel_image_1 == substrate_idx) +
             (pixel_image_2 == substrate_idx)).sum()

    # If there is no pixels with that class, IoU is undefined
    return intersection, union


def calculate_iou_metrics_run(gt_list, pred_list, classes_map):
    """Given datasets (predicted and gt) for pixel annotations, returns IoU"""
    intersection_per_substrate = defaultdict(int)
    union_per_substrate = defaultdict(int)

    for gt, pred in zip(gt_list, pred_list):
        for substrate_idx, substrate_name in classes_map.items():
            intersection, union = \
                calculate_agreement(gt, pred, substrate_idx=substrate_idx)
            if union:
                intersection_per_substrate[substrate_name] += intersection
                union_per_substrate[substrate_name] += union

    eps = 10e-16
    # Ads epsilon to the denominator to avoid division by zero
    iou_per_substrate = {
        substrate_name: intersection_per_substrate[substrate_name] /
                        (union_per_substrate[substrate_name] + eps)
        for substrate_name in classes_map.values()
    }

    iou_average = sum(intersection_per_substrate.values()) / \
                  sum(union_per_substrate.values())

    return iou_per_substrate, iou_average


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder_path', default='data', type=str,
                        help='Path to the data directory, where to save the outputs.')
    parser.add_argument('--model_path', default='models/test_model/model_best.pth', type=str,
                        help='Subfolder of data_folder_path where the images are')
    parser.add_argument('--image_folder', default='images_val', type=str,
                        help='Subfolder of data_folder_path where the images are')
    parser.add_argument('--mask_folder', default='masks_val', type=str,
                        help='Subfolder of data_folder_path where the masks are to be saved')
    parser.add_argument('--log_file_path', default=None, type=str,
                        help='Path for file to log the results')

    args = parser.parse_args()

    # Makes sure it will grab all masks and images
    masks_dir = os.listdir(os.path.join(args.data_folder_path, args.mask_folder))

    images = [x for x in os.listdir(os.path.join(args.data_folder_path, args.image_folder))]
    masks = [x.split('.')[0] + '_mask.png' for x in images]

    try:
        assert len(set(masks_dir).intersection(masks)) == len(images)
    except AssertionError:
        raise AssertionError("It seems that some images don't have masks. Re-run the create_masks script")

    print(f"Evaluating on {len(images)} images")

    # Concatenates image with full path
    images = [os.path.join(args.data_folder_path, args.image_folder, x) for x in images]
    masks = [os.path.join(args.data_folder_path, args.mask_folder, x) for x in masks]

    iou_per_substrate, iou_average = evaluate(
        data_folder_path=args.data_folder_path,
        model_path=args.model_path,
        image_file_paths=images,
        gt_file_paths=masks,
        nn_input_size=256,
        window_sizes=[500, 1000, 1500],
        step_sizes=[350, 750, 1000],
        log_file_path=args.log_file_path)
