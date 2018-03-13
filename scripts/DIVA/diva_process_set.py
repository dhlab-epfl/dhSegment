#!/usr/bin/env python
__author__ = 'solivr'

from doc_seg.loader import LoadedModel
import os
import tensorflow as tf
from glob import glob
import numpy as np
from tqdm import tqdm
import subprocess
from scipy.misc import imread, imsave
from doc_seg.post_processing.segmentation import diva_post_processing_fn
from doc_seg.evaluation.segmentation import diva_evaluate_folder
from doc_seg.evaluation.segmentation import to_original_color_code, parse_diva_tool_output
from scripts.DIVA.diva_dataset_generator import MAP_COLORS, annotate_one
import argparse


TILE_SIZE = 400


def predict_on_set(filenames_to_predict, model_dir, output_dir):
    with tf.Session():
        m = LoadedModel(model_dir, 'filename')
        for filename in tqdm(filenames_to_predict):
            pred = m.predict_with_tiles(filename, tile_size=TILE_SIZE, resized_size=None)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * pred))


export_files_test_dir = os.path.join(exported_files_eval_dir, 't06_cc50')
#os.makedirs(export_files_test_dir)
for filename in tqdm(npy_files):
    pred = np.load(filename)
    pp_diva = diva_post_processing_fn(pred/np.max(pred), thresholds=[0.6,0.6,0.6], min_cc=50)
    original_colors = to_original_color_code(np.uint8(pp_diva*255))
    # imsave('./debugs/0/{}.png'.format(os.path.basename(filename).split('.')[0]), np.uint8(pp_diva*255))
    imsave(os.path.join(export_files_test_dir, '{}_orig_colors.png'.format(os.path.basename(filename).split('.')[0])), original_colors)

filenames_exported_preds = glob(os.path.join(export_files_test_dir, '*_colors.png'))
gt_dir = '/home/datasets/DIVA/test/pixel-level-gt/'

results_list = list()
DIVA_JAR = '/home/datasets/DIVA_Layout_Analysis_Evaluator/out/artifacts/LayoutAnalysisEvaluator.jar'
for filename in tqdm(filenames_exported_preds):
    dir_name, basename = os.path.split(filename)
    label_image_filename = os.path.join(gt_dir, basename[:-16] + '.png')
    cmd = 'java -jar {} -gt {} -p {}'.format(DIVA_JAR, label_image_filename, filename)
    result = subprocess.check_output(cmd, shell=True).decode()
    results_list.append(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help='Directory of the model (should be of type ''*/export/<timestamp>)')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='Folder containing the images to evaluate the model on')
    parser.add_argument('-o', '--out_folder', type=str, required=True,
                        help='Folder containing the outputs (.npy predictions and visualization errors)')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='Which GPU to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.get('gpu')
    model_dir = args.get('model_dir')
    filenames_to_predict = glob(os.path.join(args.get('input_dir'), '*.jpg')) + \
                           glob(os.path.join(args.get('input_dir'), '*.png'))
    if len(filenames_to_predict) == 0:
        raise FileNotFoundError

    output_dir = args.get('output_dir')
    # exported_files_eval_dir = os.path.abspath(
    #     os.path.join(model_dir, '../../test', 'model_t400_{}'.format(model_dir.split('/')[-2])))
    os.makedirs(output_dir, exist_ok=True)



    npy_files = glob(os.path.join(output_dir, '*.npy'))

    export_files_test_dir = os.path.join(exported_files_eval_dir, 't06_cc50')
    # os.makedirs(export_files_test_dir)
    for filename in tqdm(npy_files):
        pred = np.load(filename)
        pp_diva = diva_post_processing_fn(pred / np.max(pred), thresholds=[0.6, 0.6, 0.6], min_cc=50)
        original_colors = to_original_color_code(np.uint8(pp_diva * 255))
        # imsave('./debugs/0/{}.png'.format(os.path.basename(filename).split('.')[0]), np.uint8(pp_diva*255))
        imsave(
            os.path.join(export_files_test_dir, '{}_orig_colors.png'.format(os.path.basename(filename).split('.')[0])),
            original_colors)
