#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
from scipy.misc import imread, imsave
import numpy as np
from tqdm import tqdm
import argparse
from typing import List

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def get_img_filenames(directory: str) -> List[str]:
    directory = os.path.abspath(directory)
    year = int(directory.split(os.path.sep)[-1].split('-')[0][-4:])
    if year == 2010:
        extension = ['*.jpg', '*.bmp', '*.tif']
    elif year == 2011 or year == 2012 or year == 2014:
        extension = ['*.png']
    elif year == 2013 or year == 2009:
        extension = ['*.bmp']
    elif year == 2016:
        extension = ['*[!_gt].bmp']
    else:
        raise NotImplementedError

    files = [glob(os.path.join(directory, ext)) for ext in extension]
    return [item for l in files for item in l]


def get_gt_filename(img_filename: str) -> str:
    directory, basename = os.path.os.path.split(img_filename)
    year = int(directory.split(os.path.sep)[-1].split('-')[0][-4:])
    if year == 2009:
        extension = '.tiff'
    elif year == 2011:
        extension = '_GT.tiff'
    elif year == 2012:
        extension = '_GT.tif'
    elif year == 2013 or year == 2014 or year == 2010:
        extension = '_estGT.tiff'
    elif year == 2016:
        extension = '_gt.bmp'
    else:
        raise NotImplementedError

    return os.path.join(directory, '{}{}'.format(basename.split('.')[0], extension))


def get_exported_image_basename(image_filename: str) -> str:
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('-')[0]
    return '{}_{}'.format(acronym, basename.split('.')[0])


def generate_one_tuple(img_filename: str, output_dir: str) -> None:
    image = imread(img_filename, mode='RGB')
    basename_export = get_exported_image_basename(img_filename)
    imsave(os.path.join(output_dir, 'images', '{}.jpg'.format(basename_export)), image)
    gt = np.zeros_like(image)
    gt[:, :, 0] = ~imread(get_gt_filename(img_filename), mode='L')
    imsave(os.path.join(output_dir, 'labels', '{}.png'.format(basename_export)), gt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, default=None,
                        help='Input directory containing images and PAGE files')
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None,
                        help='Output directory to save images and labels')
    parser.add_argument('-s', '--split', action='store_true',  # default : false
                        help='Split inputs into training and validation set')
    args = vars(parser.parse_args())

    img_filenames = get_img_filenames(args.get('input_dir'))

    if args.get('split'):
        # Split data into training and validation set (0.9/0.1)
        train_inds = np.random.choice(len(img_filenames), size=int(0.9 * len(img_filenames)), replace=False)
        train_mask = np.zeros(len(img_filenames), dtype=np.bool_)
        train_mask[train_inds] = 1
        image_filenames_train = np.array(img_filenames)[train_mask]
        image_filenames_eval = np.array(img_filenames)[~train_mask]

        # Training set
        root_train_dir = os.path.join(args.get('output_dir'), 'train')
        os.makedirs(os.path.join(root_train_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root_train_dir, 'labels'), exist_ok=True)
        for img_filename in tqdm(image_filenames_train):
            generate_one_tuple(img_filename, root_train_dir)

        # Validation set
        root_val_dir = os.path.join(args.get('output_dir'), 'validation')
        os.makedirs(os.path.join(root_val_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root_val_dir, 'labels'), exist_ok=True)
        for img_filename in tqdm(image_filenames_eval):
            generate_one_tuple(img_filename, root_val_dir)
    else:
        root_test_dir = os.path.join(args.get('output_dir'), 'test')
        os.makedirs(os.path.join(root_test_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root_test_dir, 'labels'), exist_ok=True)
        for img_filename in tqdm(img_filenames):
            generate_one_tuple(img_filename, root_test_dir)

    # Class file
    classes = np.stack([(0, 0, 0), (255, 0, 0)])
    np.savetxt(os.path.join(args.get('output_dir'), 'classes.txt'), classes, fmt='%d')