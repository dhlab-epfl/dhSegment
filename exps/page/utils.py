#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from imageio import imread, imsave
import numpy as np
import cv2
import os
from tqdm import tqdm


def get_coords_form_txt_line(line: str)-> tuple:
    """
    gets the coordinates of the page from the txt file (line-wise)
    :param line: line of the .txt file
    :return: coordinates, filename
    """
    splits = line.split(',')
    full_filename = splits[0]
    splits = splits[1:]
    if splits[-1] in ['SINGLE', 'ABNORMAL']:
        coords_simple = np.reshape(np.array(splits[:-1], dtype=int), (4, 2))
        # coords_double = None
        coords = coords_simple
    else:
        coords_simple = np.reshape(np.array(splits[:8], dtype=int), (4, 2))
        # coords_double = np.reshape(np.array(splits[-4:], dtype=int), (2, 2))
        # coords = (coords_simple, coords_double)
        coords = coords_simple

    return coords, full_filename


def make_binary_mask(txt_file):
    """
    From export txt file with filnenames and coordinates of qudrilaterals, generate binary mask of page
    :param txt_file: txt file filename
    :return:
    """
    for line in open(txt_file, 'r'):
        dirname, _ = os.path.split(txt_file)
        c, full_name = get_coords_form_txt_line(line)
        img = imread(full_name)
        label_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        label_img = cv2.fillPoly(label_img, [c[:, None, :]], 255)
        basename = os.path.basename(full_name)
        imsave(os.path.join(dirname, '{}_bin.png'.format(basename.split('.')[0])), label_img)


def page_dataset_generator(txt_filename: str, input_dir: str, output_dir: str):
    """
    Given a txt file (filename, coords corners), generates a dataset of images + labels
    :param txt_filename: File (txt) containing list of images
    :param input_dir: Root directory to original images
    :param output_dir: Output directory for generated dataset
    :return:
    """

    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for line in tqdm(open(txt_filename, 'r')):
        coords, full_filename = get_coords_form_txt_line(line)

        try:
            img = imread(os.path.join(input_dir, full_filename))
        except FileNotFoundError:
            print('File {} not found'.format(full_filename))
            continue
        label_img = np.zeros((img.shape[0], img.shape[1], 3))

        label_img = cv2.fillPoly(label_img, [coords], (255, 0, 0))
        # if coords_double is not None:
        #     label_img = cv2.polylines(label_img, [coords_double], False, color=(0, 0, 0), thickness=50)

        col, filename = full_filename.split(os.path.sep)[-2:]

        imsave(os.path.join(output_img_dir, '{}_{}.jpg'.format(col.split('_')[0], filename.split('.')[0])), img)
        imsave(os.path.join(output_label_dir, '{}_{}.png'.format(col.split('_')[0], filename.split('.')[0])), label_img)

    # Class file
    classes = np.stack([(0, 0, 0), (255, 0, 0)])
    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')
