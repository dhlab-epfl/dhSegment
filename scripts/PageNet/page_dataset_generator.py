#!/usr/bin/env python
__author__ = 'solivr'

from scipy.misc import imread, imsave
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str, help='File (txt) containing list of images')
    parser.add_argument('-i', '--input_dir', required=True, type=str, help='Root directory to original images')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output directory for generated dataset')
    args = vars(parser.parse_args())

    output_img_dir = os.path.join(args.get('output_dir'), 'images')
    output_label_dir = os.path.join(args.get('output_dir'), 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for line in tqdm(open(args.get('file'), 'r')):
        line = line[:-1]
        splits = line.split(',')
        full_filename = splits[0]
        splits = splits[1:]
        if splits[-1] in ['SINGLE', 'ABNORMAL']:
            coords_simple = np.reshape(np.array(splits[:-1], dtype=int), (4, 2))
            coords_double = None
            coords = coords_simple
        else:
            coords_simple = np.reshape(np.array(splits[:8], dtype=int), (4, 2))
            coords_double = np.reshape(np.array(splits[-4:], dtype=int), (2, 2))
            coords = (coords_simple, coords_double)

        img = imread(os.path.join(args.get('input_dir'), full_filename))
        label_img = np.zeros((img.shape[0], img.shape[1], 3))

        label_img = cv2.fillPoly(label_img, [coords_simple], (255, 0, 0))
        if coords_double is not None:
            label_img = cv2.polylines(label_img, [coords_double], False, color=(0, 0, 0), thickness=50)

        col, filename = full_filename.split(os.path.sep)[-2:]

        imsave(os.path.join(output_img_dir, '{}_{}.jpg'.format(col.split('_')[0], filename.split('.')[0])), img)
        imsave(os.path.join(output_label_dir, '{}_{}.png'.format(col.split('_')[0], filename.split('.')[0])), label_img)

    # Class file
    classes = np.stack([(0, 0, 0), (255, 0, 0)])
    np.savetxt(os.path.join(args.get('output_dir'), 'classes.txt'), classes, fmt='%d')
