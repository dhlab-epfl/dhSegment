#!/usr/bin/env python
__author__ = 'solivr'

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from doc_seg.loader import LoadedModel
from doc_seg.post_processing import boxes_detection
from exps.Page.page_post_processing import page_post_processing_fn
from exps.evaluation.base import format_quad_to_string
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse
from glob import glob
from scipy.misc import imsave, imread
import tempfile
import cv2
import json


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


def find_page(img_filenames, dir_predictions, post_process_params, output_dir, mode='quadrilateral', debug=False):
    """

    :param img_filenames:
    :param dir_predictions:
    :param post_process_params:
    :param output_dir:
    :return:
    """

    with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        for filename in tqdm(img_filenames, 'Post-processing'):
            orig_img = imread(filename, mode='RGB')
            basename = os.path.basename(filename).split('.')[0]

            filename_pred = os.path.join(dir_predictions, basename + '.npy')
            pred = np.load(filename_pred)
            page_bin = page_post_processing_fn(pred / np.max(pred), **post_process_params)

            target_shape = (orig_img.shape[1], orig_img.shape[0])
            bin_upscaled = cv2.resize(np.uint8(page_bin), target_shape, interpolation=cv2.INTER_NEAREST)
            if debug:
                imsave(os.path.join(output_dir, '{}_bin.png'.format(basename)), bin_upscaled)

            if mode == 'no_box':
                _, contours, _ = cv2.findContours(np.uint8(bin_upscaled), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_cnt = cv2.polylines(orig_img.copy(), contours, True, (0, 0, 255), thickness=15)
                imsave(os.path.join(output_dir, '{}_cnt.jpg'.format(basename)), img_cnt)
                img_mask = cv2.fillPoly(np.zeros((orig_img.shape[:2]), np.uint8), contours, 255)
                imsave(os.path.join(output_dir, '{}_mask.png'.format(basename)), img_mask)
            else:
                pred_box = boxes_detection.find_box(np.uint8(bin_upscaled), mode=mode)
                if pred_box is not None:
                    cv2.polylines(orig_img, [pred_box[:, None, :]], True, (0, 0, 255), thickness=15)
                else:
                    print('No box found in {}'.format(filename))
                imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), orig_img)

                f.write('{},{}\n'.format(filename, format_quad_to_string(pred_box)))


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
    parser.add_argument('-b', '--box_mode', type=str, default='quadrilateral',
                        help="Which type of box to use 'quadrilateral', 'min_rectangle', 'rectangle', 'no_box'")
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

    if post_process_params:
        with open(post_process_params, 'r') as f:
            post_process_params = json.load(f)
        post_process_params = post_process_params['params']
    else:
        post_process_params = {"threshold": -1, "ksize_open": [5, 5], "ksize_close": [5, 5]}

    # Prediction
    with tempfile.TemporaryDirectory() as tmpdirname:
        npy_directory = tmpdirname
        predict_on_set(input_files, model_dir, npy_directory)

        npy_files = glob(os.path.join(npy_directory, '*.npy'))
        find_page(input_files, npy_directory, post_process_params, output_dir, mode=args.get('box_mode'))



