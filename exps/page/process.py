#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from dh_segment.inference import LoadedModel
from imageio import imsave
from dh_segment.post_processing import binarization
from dh_segment.post_processing.boxes_detection import find_boxes


def prediction_fn(model_dir: str, input_dir: str, output_dir: str=None) -> None:
    """
    Given a model directory this function will load the model and apply it to the files (.jpg, .png) found in input_dir.
    The predictions will be saved in output_dir as .npy files (values ranging [0,255])
    :param model_dir: Directory containing the saved model
    :param input_dir: input directory where the images to predict are
    :param output_dir: output directory to save the predictions (probability images)
    :return:
    """
    if not output_dir:
        # For model_dir of style model_name/export/timestamp/ this will create a folder model_name/predictions'
        output_dir = '{}'.format(os.path.sep).join(model_dir.split(os.path.sep)[:-3] + ['predictions'])

    os.makedirs(output_dir, exist_ok=True)
    filenames_to_predict = glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.png'))
    # Load model
    with tf.Session():
        m = LoadedModel(model_dir, predict_mode='filename_original_shape')
        for filename in tqdm(filenames_to_predict, desc='Prediction'):
            pred = m.predict(filename)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]), np.uint8(255 * pred))


def page_post_processing_fn(probs: np.ndarray, threshold: float=0.5, output_basename: str=None,
                            kernel_size: int = 5) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :param output_basename:
    :param kernel_size: size of kernel for morphological cleaning
    """

    mask = binarization.thresholding(probs[:, :, 1], threshold=threshold)
    result = binarization.cleaning_binary(mask, size=kernel_size)

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)
    return result


def format_quad_to_string(quad):
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


def extract_page(prediction: np.ndarray, min_area: float=0.2, post_process_params: dict=None) -> list():
    """
    Given an image with probabilities, post-processes it and extracts one box
    :param prediction: probability mask [0, 1]
    :param min_area: minimum area to be considered as a valid extraction
    :param post_process_params: params for page prost processing function
    :return: list of coordinates of boxe
    """
    if post_process_params:
        post_pred = page_post_processing_fn(prediction, **post_process_params)
    else:
        post_pred = prediction
    pred_box = find_boxes(np.uint8(post_pred), mode='quadrilateral', min_area=min_area, n_max_boxes=1)

    return pred_box
