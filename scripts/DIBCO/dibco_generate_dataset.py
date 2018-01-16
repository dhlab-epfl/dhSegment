#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
from scipy.misc import imread, imsave
import numpy as np
from tqdm import tqdm
import cv2
import argparse


TARGET_HEIGHT = 800


def get_img_filenames(directory):
    directory = os.path.abspath(directory)
    year = int(directory.split(os.path.sep)[-1].split('-')[0][-4:])
    if year == 2011 or year == 2012 or year == 2014:
        extension = '*.png'
    elif year == 2013:
        extension = '*.bmp'
    elif year == 2016:
        extension = '*[!_gt].bmp'
    else:
        raise NotImplementedError

    return glob(os.path.join(directory, extension))


def get_gt_filename(img_filename):
    directory, basename = os.path.os.path.split(img_filename)
    year = int(directory.split(os.path.sep)[-1].split('-')[0][-4:])
    if year == 2011:
        extension = 'GT.tiff'
    elif year == 2012:
        extension = 'GT.tif'
    elif year == 2013 or year == 2014:
        extension = 'estGT.tiff'
    elif year == 2016:
        extension = 'gt.bmp'
    else:
        raise NotImplementedError

    return os.path.join(directory, '{}_{}'.format(basename.split('.')[0], extension))


def save_and_resize(img, filename, nearest=False):
    resized = cv2.resize(img, ((img.shape[1]*TARGET_HEIGHT)//img.shape[0], TARGET_HEIGHT),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    imsave(filename, resized)


def get_exported_image_basename(image_filename):
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('-')[0]
    return '{}_{}'.format(acronym, basename.split('.')[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, default=None,
                        help='Input directory containing images and PAGE files')
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None,
                        help='Output directory to save images and labels')
    parser.add_argument('-th', '--target_height', type=int, default=800,
                        help='The desired output image height (in px)')
    args = vars(parser.parse_args())

    os.makedirs(os.path.join(args.get('output_dir'), 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.get('output_dir'), 'labels'), exist_ok=True)

    img_filenames = get_img_filenames(args.get('input_dir'))

    for img_filename in tqdm(img_filenames):
        image = imread(img_filename)
        basename_export = get_exported_image_basename(img_filename)
        save_and_resize(image,
                        os.path.join(args.get('output_dir'), 'images', '{}.jpg'.format(basename_export)))
        gt = np.zeros_like(image)
        gt[:, :, 0] = ~imread(get_gt_filename(img_filename), mode='L')
        save_and_resize(gt,
                        os.path.join(args.get('output_dir'), 'labels', '{}.png'.format(basename_export)),
                        nearest=True)
