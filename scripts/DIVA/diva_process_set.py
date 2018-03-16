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
from scipy.misc import imsave
from doc_seg.post_processing.segmentation import diva_post_processing_fn
from doc_seg.evaluation.segmentation import to_original_color_code, parse_diva_tool_output
import argparse


TILE_SIZE = 400
DIVA_JAR = '/home/datasets/DIVA_Layout_Analysis_Evaluator/out/artifacts/LayoutAnalysisEvaluator.jar'
PARAMS = {'thresholds': [0.6, 0.6, 0.6], 'min_cc': 100}


def predict_on_set(filenames_to_predict, model_dir, output_dir):
    with tf.Session():
        m = LoadedModel(model_dir, 'filename')
        for filename in tqdm(filenames_to_predict, desc='Prediction'):
            pred = m.predict_with_tiles(filename, tile_size=TILE_SIZE, resized_size=None)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * pred))


def evaluate_on_set(files_to_evaluate, post_process_params, output_dir, gt_dir):
    results_list = list()
    output_dir = os.path.join(output_dir, 'th{}cc{}'.format(post_process_params['thresholds'][0],
                                                         post_process_params['min_cc']))
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(files_to_evaluate, 'Post-process and evaluation'):
        pred = np.load(filename)
        pp_diva = diva_post_processing_fn(pred / np.max(pred), **post_process_params)
        original_colors = to_original_color_code(np.uint8(pp_diva * 255))
        # imsave('./debugs/0/{}.png'.format(os.path.basename(filename).split('.')[0]), np.uint8(pp_diva*255))
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
    parser.add_argument('-e', '--eval_only', default=False, action='store_true',
                        help='Whether to make or not the prediction')
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

    # TODO Get params from file
    gt_dir = args.get('ground_truth_dir')
    mean_iu = evaluate_on_set(npy_files, PARAMS, output_dir, gt_dir)
    print('MEAN IU : {}'.format(mean_iu))