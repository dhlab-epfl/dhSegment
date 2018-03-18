#!/usr/bin/env python
__author__ = 'solivr'

from scipy.misc import imread, imsave
import numpy as np
from tqdm import tqdm
from glob import glob
import os
from collections import OrderedDict
import argparse

MAP_COLORS = OrderedDict([('background', ((0, 0, 1), (0, 0, 0))),
                          ('comment', ((0, 0, 2), (0, 0, 255))),
                          ('decoration', ((0, 0, 4), (0, 255, 0))),
                          ('text', ((0, 0, 8), (255, 0, 0))),
                          ('comment_deco', ((0, 0, 6), (0, 255, 255))),
                          ('text_comment', ((0, 0, 10), (255, 0, 255))),
                          ('text_deco', ((0, 0, 12), (255, 255, 0)))])


def annotate_one(gt_image: np.array, map_colors: dict):
    label_img = np.zeros_like(gt_image)
    for key, colors in map_colors.items():
        label_img[np.all(gt_image == colors[0], axis=-1)] = colors[1]

    return label_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, default=None,
                        help='Input directory containing images and PAGE files')
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None,
                        help='Output directory to save images and labels')
    args = vars(parser.parse_args())

    img_filenames = glob(os.path.join(args.get('input_dir'), 'img', '*.jpg'))
    output_img_dir = os.path.join(args.get('output_dir'), 'images')
    output_label_dir = os.path.join(args.get('output_dir'), 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for filename in tqdm(img_filenames):
        img = imread(filename, mode='RGB')
        basename = os.path.basename(filename).split('.')[0]
        filename_label = os.path.join(args.get('input_dir'), 'pixel-level-gt', '{}.png'.format(basename))
        gt_img = imread(filename_label, mode='RGB')
        label_image = annotate_one(gt_img, MAP_COLORS)

        # Save
        imsave(os.path.join(output_img_dir, '{}.jpg'.format(basename)), img)
        imsave(os.path.join(output_label_dir, '{}.png'.format(basename)), label_image)

    # # Class file
    # classes = np.array([0, 0, 0])
    # for c in MAP_COLORS.values():
    #     classes = np.vstack([classes, np.array(c[1])])
    #
    # codes_list = list()
    # n_bits = len('{:b}'.format(classes.shape[0]))
    # for i in range(classes.shape[0]):
    #     codes_list.append('{:08b}'.format(i)[-n_bits:])
    # codes_ints = [[int(char) for char in code] for code in codes_list]
    # classes = np.hstack((classes, np.array(codes_ints)))
    #
    # np.savetxt(os.path.join(args.get('output_dir'), 'classes.txt'), classes, fmt='%d')
