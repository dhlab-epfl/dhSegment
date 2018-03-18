#!/usr/bin/env python
__author__ = 'solivr'

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from doc_seg.loader import LoadedModel
from doc_seg.post_processing.line_detection import line_extraction_v1

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
from glob import glob
from scipy.misc import imsave, imread
import tempfile
import json
from doc_seg.post_processing import PAGE


def predict_on_set(filenames_to_predict, model_dir, output_dir):
    """

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


def find_lines(img_filenames, dir_predictions, post_process_params, output_dir, debug=False):
    """

    :param img_filenames:
    :param dir_predictions:
    :param post_process_params:
    :param output_dir:
    :return:
    """

    for filename in tqdm(img_filenames, 'Post-processing'):
        orig_img = imread(filename, mode='RGB')
        basename = os.path.basename(filename).split('.')[0]

        filename_pred = os.path.join(dir_predictions, basename + '.npy')
        pred = np.load(filename_pred)

        contours, lines_mask = line_extraction_v1(pred[:, :, 1], **post_process_params)
        if debug:
            imsave(os.path.join(output_dir, '{}_bin.jpg'.format(basename)), lines_mask)

        ratio = (orig_img.shape[0] / pred.shape[0], orig_img.shape[1] / pred.shape[1])
        xml_filename = os.path.join(output_dir, basename + '.xml')
        PAGE.save_baselines(xml_filename, contours, ratio, initial_shape=pred.shape[:2])

        generated_page = PAGE.parse_file(xml_filename)
        generated_page.draw_baselines(orig_img, color=(0, 0, 255))
        imsave(os.path.join(output_dir, '{}_lines.jpg'.format(basename)), orig_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='Directory of the model (should be of type ''*/export/<timestamp>)')
    parser.add_argument('-i', '--input_files', type=str, required=True, nargs='+',
                        help='Folder containing the images to evaluate the model on')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Folder containing the outputs (.npy predictions and visualization errors)')
    parser.add_argument('--post_process_params', type=str, default=None,
                        help='JSOn file containing the params for post-processing')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    # parser.add_argument('-pp', '--post_proces_only', default=False, action='store_true',
    #                     help='Whether to make or not the prediction')
    args = parser.parse_args()
    args = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.get('gpu')
    model_dir = args.get('model_dir')
    input_files = args.get('input_files')
    if len(input_files) == 0:
        raise FileNotFoundError

    output_dir = args.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
    post_process_params = args.get('post_proces_params')

    if post_process_params is not None:
        with open(post_process_params, 'r') as f:
            post_process_params = json.load(f)
        post_process_params = post_process_params['params']
    else:
        post_process_params = {"low_threshold": 0.1, "sigma": 2.5, "high_threshold": 0.3}

    # Prediction
    with tempfile.TemporaryDirectory() as tmpdirname:
        npy_directory = tmpdirname
        predict_on_set(input_files, model_dir, npy_directory)

        npy_files = glob(os.path.join(npy_directory, '*.npy'))

        find_lines(input_files, npy_directory, post_process_params, output_dir, debug=True)



