#!/usr/bin/env python

import tensorflow as tf
from dh_segment.loader import LoadedModel
from dh_segment.post_processing import boxes_detection
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import cv2
from scipy.misc import imread, imsave


def page_post_processing_fn(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :param ksize_open: size of kernel for morphological opening
    :param ksize_close: size of kernel for morphological closing
    :return: binary mask
    """
    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        blur = cv2.GaussianBlur(probs, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.uint8(bin_img / 255)
    else:
        mask = probs > threshold

    ksize_open = (5, 5)
    ksize_close = (5, 5)
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones(ksize_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones(ksize_close))

    return mask / 255


def format_quad_to_string(quad):
    """
    Formats the corner points into a string.
    :param quad: coordinates of the quadrilateral
    :return:
    """
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


def find_page(filename, prediction):
    """
    Finds the rectangle enclosing the page and writes the coordinates into a txt file. It also exports the images
    with the page drawn.
    :param filename: filename of the image to process
    :param prediction: probability map output by the model
    :param output_dir: directory to output the txt file with the corner points of the page and the images with
            the highlighted page
    :return:
    """

    # Load original image
    orig_img = imread(filename, mode='RGB')

    # Make binary mask
    page_bin = page_post_processing_fn(prediction)

    # Upscale to have full resolution image
    target_shape = (orig_img.shape[1], orig_img.shape[0])
    bin_upscaled = cv2.resize(np.uint8(page_bin), target_shape, interpolation=cv2.INTER_NEAREST)

    # Find quadrilateral enclosing the page
    pred_box = boxes_detection.find_boxes(np.uint8(bin_upscaled), mode='min_rectangle')

    return pred_box, orig_img


if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    export_models_dir = glob('demo/page_model/export/*')
    if not export_models_dir:
        model_dir = 'demo/model/'
    else:
        export_models_dir.sort()
        model_dir = export_models_dir[-1]

    input_files = glob('demo/pages/test_a1/images/*')

    output_dir = 'demo/processed_images'
    os.makedirs(output_dir, exist_ok=True)

    # Store coordinates of page in a .txt file
    txt_coordinates = ''

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, 'filename')

        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            pred = m.predict(filename)['probs'][0]
            pred = pred[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
            pred = pred / np.max(pred)  # Normalize to be in [0, 1]

            # Find page coordinates
            pred_page, original_img = find_page(filename, pred)

            # Draw page box on image and export it. Add also box coordinates to the txt file
            if pred_page is not None:
                cv2.polylines(original_img, [pred_page[:, None, :]], True, (0, 0, 255), thickness=5)
                # Write corners points into a .txt file
                txt_coordinates += '{},{}\n'.format(filename, format_quad_to_string(pred_page))
            else:
                print('No box found in {}'.format(filename))
            basename = os.path.basename(filename).split('.')[0]
            imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), original_img)

    # Save txt file
    with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        f.write(txt_coordinates)
