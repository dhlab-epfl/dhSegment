#!/usr/bin/env python
__author__ = 'solivr'

from .page_dataset_generator import get_coords_form_txt_line
from scipy.misc import imread, imsave
import numpy as np
import cv2
import os


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
