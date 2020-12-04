from collections import defaultdict
import os
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


def evaluate(image_file_paths, gt_file_paths, model_path,
             nn_input_size, window_sizes=None,
             step_sizes=None,
             device=None, data_folder_path=None):
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

    #Calculates IoU looping over images and masks

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
