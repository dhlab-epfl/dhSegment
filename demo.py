#!/usr/bin/env python

import tensorflow as tf
from dh_segment.loader import LoadedModel
from dh_segment.post_processing import boxes_detection
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import cv2
import tempfile
from scipy.misc import imread, imsave


def predict_on_set(filenames_to_predict, model_dir, output_dir):
    """
    Given a list of filenames, loads the model and saves the probabilities into a .npy file
    :param filenames_to_predict:
    :param model_dir:
    :param output_dir:
    :return:
    """
    with tf.Session():
        m = LoadedModel(model_dir, 'filename')
        for filename in tqdm(filenames_to_predict, desc='Prediction'):
            pred = m.predict(filename)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * pred))


def page_post_processing_fn(probs: np.ndarray, threshold: float=-1, ksize_open: tuple=(5, 5),
                            ksize_close: tuple=(5, 5)) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :param ksize_open: size of kernel for morphological opening
    :param ksize_close: size of kernel for morphological closing
    :return: binary mask
    """
    probs = probs[:, :, 1]
    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        blur = cv2.GaussianBlur(probs, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.uint8(bin_img / 255)
    else:
        mask = probs > threshold
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones(ksize_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones(ksize_close))

    result = mask / 255

    return result


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


def find_page(img_filenames, dir_predictions, output_dir):
    """
    Finds the rectangle enclosing the page and writes the coordinates into a txt file. It also exports the images
    with the page drawn.
    :param img_filenames: list of filenames to process
    :param dir_predictions: directory where are stored the temporary .npy files (probabilities maps) output by the model
    :param output_dir: directory to output the txt file with the corner points of the page and the images with
            the highlighted page
    :return:
    """

    with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        for filename in tqdm(img_filenames, 'Post-processing'):
            orig_img = imread(filename, mode='RGB')
            basename = os.path.basename(filename).split('.')[0]

            filename_pred = os.path.join(dir_predictions, basename + '.npy')
            pred = np.load(filename_pred)
            # Make binary mask
            page_bin = page_post_processing_fn(pred / np.max(pred))

            # Upscale to have full resolution image
            target_shape = (orig_img.shape[1], orig_img.shape[0])
            bin_upscaled = cv2.resize(np.uint8(page_bin), target_shape, interpolation=cv2.INTER_NEAREST)

            # Find quadrilateral enclosing the page
            pred_box = boxes_detection.find_boxes(np.uint8(bin_upscaled), mode='quadrilateral')

            if pred_box is not None:
                cv2.polylines(orig_img, [pred_box[:, None, :]], True, (0, 0, 255), thickness=5)
            else:
                print('No box found in {}'.format(filename))
            imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), orig_img)

            # Write corners points into a .txt file
            f.write('{},{}\n'.format(filename, format_quad_to_string(pred_box)))


if __name__ == '__main__':

    export_models_dir = glob('demo/page_model/export/*')
    if not export_models_dir:
        model_dir = 'demo/model/'
    else:
        export_models_dir.sort()
        model_dir = export_models_dir[-1]

    input_files = glob('demo/pages/test_a1/images/*')

    output_dir = 'demo/processed_images'
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        npy_directory = tmpdirname
        # Load model and output probabilities masks
        predict_on_set(input_files, model_dir, npy_directory)

        npy_files = glob(os.path.join(npy_directory, '*.npy'))
        find_page(input_files, npy_directory, output_dir)
