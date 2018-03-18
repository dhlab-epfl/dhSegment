#!/usr/bin/env python
__author__ = 'solivr'

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from doc_seg.loader import LoadedModel
import tensorflow as tf
from glob import glob
import numpy as np
from tqdm import tqdm
import subprocess
from scipy.misc import imsave, imread
import cv2
from diva_post_processing import diva_post_processing_fn
from diva_evaluation import to_original_color_code, parse_diva_tool_output
import argparse
import json


TILE_SIZE = 400
DIVA_JAR = '/home/datasets/DIVA_Layout_Analysis_Evaluator/out/artifacts/LayoutAnalysisEvaluator.jar'
PARAMS = {'thresholds': [0.5, 0.5, 0.5], 'min_cc': 50}


def predict_on_set(filenames_to_predict, model_dir, output_dir):
    with tf.Session():
        m = LoadedModel(model_dir, 'filename')
        for filename in tqdm(filenames_to_predict, desc='Prediction'):
            pred = m.predict_with_tiles(filename, tile_size=TILE_SIZE, resized_size=None)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * pred))


def evaluate_on_set(files_to_evaluate, post_process_params, output_dir, gt_dir, page_masks_dir=None):
    results_list = list()
    output_dir = os.path.join(output_dir, 'th{}cc{}_{}'.format(post_process_params['thresholds'][0],
                                                         post_process_params['min_cc'], np.random.randint(0, 10e4)))
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(files_to_evaluate, 'Post-process and evaluation'):
        if page_masks_dir is not None:
            page_mask = imread(os.path.join(page_masks_dir,
                                            '{}_bin.png'.format(os.path.basename(filename).split('.')[0])), mode='L')
            # page_mask = cv2.morphologyEx(page_mask, cv2.MORPH_ERODE, kernel=np.ones((45, 45)))
        else:
            page_mask = None

        preds = np.load(filename)
        pp_diva = diva_post_processing_fn(preds / np.max(preds), page_mask=page_mask, **post_process_params)
        original_colors = to_original_color_code(np.uint8(pp_diva * 255))
        pred_img_filename = os.path.join(output_dir,
                                         '{}_orig_colors.png'.format(os.path.basename(filename).split('.')[0]))
        imsave(pred_img_filename, original_colors)

        dir_name, basename = os.path.split(pred_img_filename)
        label_image_filename = os.path.join(gt_dir, basename[:-16] + '.png')
        cmd = 'java -jar {} -gt {} -p {}'.format(DIVA_JAR, label_image_filename, pred_img_filename)
        result = subprocess.check_output(cmd, shell=True).decode()
        results_list.append(result)

    miu = list()
    with open(os.path.join(output_dir, 'raw_results.txt'), 'w') as f:
        f.write('Post-process params : {}\n'.format(post_process_params))
        f.write('Page mask : {}\n'.format(page_masks_dir))
        for res in results_list:
            r = parse_diva_tool_output(res)
            miu.append(r['Mean_IU'])
            f.write(res)
        f.write('--- Mean IU : {}'.format(np.mean(miu)))
    return np.mean(miu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='Directory of the model (should be of type ''*/export/<timestamp>)')
    parser.add_argument('-i', '--input_files', type=str, required=True, nargs='+',
                        help='Folder containing the images to evaluate the model on')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Folder containing the outputs (.npy predictions and visualization errors)')
    parser.add_argument('-gt', '--ground_truth_dir', type=str, required=True,
                        help='Ground truth directory containing the abeled images')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    parser.add_argument('-p', '--params_file', type=str, default=None,
                        help='JSON params file')
    parser.add_argument('-e', '--eval_only', default=False, action='store_true',
                        help='Whether to make or not the prediction')
    parser.add_argument('-pm', '--page_masks_dir', type=str, default=None,
                        help='Directory containing the binary masks of extracted pages')
    args = parser.parse_args()
    args = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.get('gpu')
    model_dir = args.get('model_dir')
    input_files = args.get('input_files')
    if len(input_files) == 0:
        raise FileNotFoundError

    output_dir = args.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    # Prediction
    if not args.get('eval_only'):
        predict_on_set(input_files, model_dir, output_dir)
    npy_files = glob(os.path.join(output_dir, '*.npy'))

    if args.get('params_file') is None:
        print('No params file found')
        params_list = [PARAMS]
    else:
        with open(args.get('params_file'), 'r') as f:
            configs_data = json.load(f)
            # If the file contains a list of configurations
            if 'configs' in configs_data.keys():
                params_list = configs_data['configs']
                assert isinstance(params_list, list)
            # Or if there is a single configuration
            else:
                params_list = [configs_data]

    gt_dir = args.get('ground_truth_dir')
    for params in tqdm(params_list, desc='Params'):
        print(params)
        mean_iu = evaluate_on_set(npy_files, params, output_dir, gt_dir, args.get('page_masks_dir'))
        print('MEAN IU : {}'.format(mean_iu))