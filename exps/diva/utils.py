#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from glob import glob
from imageio import imread, imsave
import os


MAP_COLORS = OrderedDict([('background', ((0, 0, 1), (0, 0, 0))),
                          ('comment', ((0, 0, 2), (0, 0, 255))),
                          ('decoration', ((0, 0, 4), (0, 255, 0))),
                          ('text', ((0, 0, 8), (255, 0, 0))),
                          ('comment_deco', ((0, 0, 6), (0, 255, 255))),
                          ('text_comment', ((0, 0, 10), (255, 0, 255))),
                          ('text_deco', ((0, 0, 12), (255, 255, 0)))])


def parse_diva_tool_output(score_txt: str, output_json_filename: str=None)-> dict:
    """
    This fn parses the output of JAR DIVA Evaluation tool
    :param score_txt: filename of txt score containing output of DIVA evaluation tool
    :param output_json_filename: filename to output the parsed result in json
    :return: dict containing the parsed results
    """
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

    if output_json_filename is not None:
        with open(output_json_filename, 'w') as f:
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


def diva_dataset_generator(input_dir: str, output_dir: str):
    """

    :param input_dir: Input directory containing images and PAGE files
    :param output_dir: Output directory to save images and labels
    :return:
    """

    img_filenames = glob(os.path.join(input_dir, 'img', '*.jpg'))
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    def annotate_one(gt_image: np.array, map_colors: dict=MAP_COLORS):
        label_img = np.zeros_like(gt_image)
        for key, colors in map_colors.items():
            label_img[np.all(gt_image == colors[0], axis=-1)] = colors[1]

        return label_img

    for filename in tqdm(img_filenames):
        img = imread(filename, pilmode='RGB')
        basename = os.path.basename(filename).split('.')[0]
        filename_label = os.path.join(input_dir, 'pixel-level-gt', '{}.png'.format(basename))
        gt_img = imread(filename_label, pilmode='RGB')
        label_image = annotate_one(gt_img, MAP_COLORS)

        # Save
        imsave(os.path.join(output_img_dir, '{}.jpg'.format(basename)), img)
        imsave(os.path.join(output_label_dir, '{}.png'.format(basename)), label_image)
