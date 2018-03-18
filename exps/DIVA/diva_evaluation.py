#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imsave
import subprocess
import numpy as np
import cv2
import json
from diva_dataset_generator import MAP_COLORS


DIVA_CLASSES = {
    -1: 'background',
    0: 'main_text',
    1: 'decorations',
    2: 'comments'
                }
DIVA_JAR = '/home/datasets/DIVA_Layout_Analysis_Evaluator/out/artifacts/LayoutAnalysisEvaluator.jar'
DIVA_VIZ_COLORS = {
    # GREEN: Foreground predicted correctly
    # YELLOW: Foreground predicted - but the wrong class (e.g. Text instead of Comment)
    # BLACK: Background predicted correctly
    # RED: Background mis-predicted as Foreground
    # BLUE: Foreground mis-predicted as Background
    'YELLOW': (255, 255, 0),
    'GREEN': (0, 127, 0),  # (TP)
    'RED': (255, 0, 0),
    'CYAN': (0, 255, 255),
    'BLACK': (0, 0, 0)  # (TP)
}


def diva_evaluate_folder(output_folder: str, validation_dir: str, diva_jar: str=DIVA_JAR,
                         debug_folder: str = None, verbose: bool = False) -> dict:
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    filenames_post_processed = glob(os.path.join(output_folder, '*.png'))

    mean_IU, mean_f1, mean_recall, mean_precision, mean_accuracy = list(), list(), list(), list(), list()
    for filename in tqdm(filenames_post_processed, desc='Evaluation'):
        basename = os.path.basename(filename).split('.')[0]

        # Open post_processed and label image
        post_processed_img = imread(filename)

        label_image_filename = os.path.join(validation_dir, 'labels', '{}.png'.format(basename))
        label_image = imread(label_image_filename)

        # Upsample processed image to compare it to original image
        if post_processed_img.shape[1] != label_image.shape[1] and post_processed_img.shape[0] != label_image.shape[0]:
            target_shape = (label_image.shape[1], label_image.shape[0])
            bin_upscaled = cv2.resize(np.uint8(post_processed_img), target_shape, interpolation=cv2.INTER_NEAREST)
        else:
            bin_upscaled = post_processed_img

        if debug_folder is not None:
            imsave(os.path.join(debug_folder, basename + '_up.png'), bin_upscaled)
            # imsave(os.path.join(debug_folder, basename + '_label.png'), label_image)

        # Save upscaled image in tmp folder :
        prediction_filename = os.path.join(output_folder, basename + '.png')
        converted_pred = to_original_color_code(bin_upscaled)
        imsave(prediction_filename, converted_pred)
        cmd = 'java -jar {} -gt {} -p {}'.format(diva_jar, label_image_filename, prediction_filename)
        result = subprocess.check_output(cmd, shell=True).decode()
        if debug_folder is not None:
            with open(os.path.join(debug_folder, '{}_scores.txt'.format(basename)), 'w') as f:
                f.write(result)
        results_json = parse_diva_tool_output(result, os.path.join(output_folder,
                                                                   '{}_scores.json'.format(basename)))

        mean_IU.append(results_json['Mean_IU'])
        mean_f1.append(results_json['F1'])
        mean_recall.append(results_json['R'])
        mean_precision.append(results_json['P'])

        # Global accuracy on the final .visulaization.png image, not accuracy per class
        viz_img = imread(prediction_filename.split('.')[0] + '.visualization.png')
        mask_tp = np.logical_or(np.all(viz_img == DIVA_VIZ_COLORS['GREEN'], axis=-1),
                                np.all(viz_img == DIVA_VIZ_COLORS['BLACK'], axis=-1))
        mean_accuracy.append(np.sum(mask_tp)/np.prod(viz_img.shape[:2]))

    return {
        'precision': np.mean(mean_precision),
        'recall': np.mean(mean_recall),
        'IU': np.mean(mean_IU),
        'f_measure': np.mean(mean_f1),
        'accuracy': np.mean(mean_accuracy)  # GLOBAL accuracy
    }


def parse_diva_tool_output(score_txt, output_json=None):
    def process_hlp_fn(string):
        """
        Processes format : R=0.64,0.52
        """
        key, vals = string.split('=')
        vals = vals.split(',')
        return {key: float(vals[0]), key + '_fw': float(vals[1])}, key

    def process_hlp_per_class_format(string, measure_key):
        """
        Processes format : '0.26|0.77|0.77|0.77
        '"""
        return {measure_key: [float(t) for t in string.split('|')]}

    lines = score_txt.splitlines()

    dic_results = {'Mean_IU': float(lines[0].split(' = ')[1])}
    measures = lines[1].split(' ')
    dic_results = {**dic_results, **{m.split('=')[0]: float(m.split('=')[1]) for m in measures[:2]}}
    for m in measures[2:5]:
        eq, tab = m.split('[')
        dic, key = process_hlp_fn(eq)
        dic_results = {**dic_results, **dic}
        dic_results = {**dic_results, **process_hlp_per_class_format(tab[:-1], key + '_per_class')}
    sp = measures[-1].split('[')
    dic, key = process_hlp_fn(sp[0])
    dic_results = {**dic_results, **dic}
    dic_results = {**dic_results, **process_hlp_per_class_format(sp[1][:-6], key + '_per_class')}
    dic_results = {**dic_results, **process_hlp_per_class_format(sp[2][:-1], sp[1][-5:-1])}

    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(dic_results, f)

    return dic_results


def to_original_color_code(bin_prediction):
    """
    (0,0,0) : Background
    (255,0,0) : Text
    (0,255,0) : decoration
    (0,0,255) : comment

    RGB=0x000008: main text body
    RGB=0x000004: decoration
    RGB=0x000002: comment
    RGB=0x000001: background
    RGB=0x00000A: main text body+comment
    RGB=0x00000C: main text body+decoration
    RGB=0x000006: comment +decoration

    :param bin_prediction:
    :return:
    """
    pred_original_colors = np.zeros_like(bin_prediction)
    for key, colors in MAP_COLORS.items():
        pred_original_colors[np.all(bin_prediction == colors[1], axis=-1)] = colors[0]

    return pred_original_colors
