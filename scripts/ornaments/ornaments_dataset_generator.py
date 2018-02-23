#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
from sklearn.model_selection import train_test_split
import argparse
from scipy.misc import imsave, imread
from tqdm import tqdm
import numpy as np


INPUT_FOLDER = '/home/datasets/ornaments/sets_all_ornaments/all_pages/images/'


def generate_set(filenames, output_dir, set: str):
    """

    :param filenames:
    :param output_dir:
    :param set: Should be 'train', 'test' or 'validation'
    :return:
    """

    out_dir = os.path.join(output_dir, set)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    for file in tqdm(filenames, desc='Generated files'):
        basename = os.path.split(file)[1].split('.')[0]
        imsave(os.path.join(out_dir, 'images', '{}.jpg'.format(basename)), imread(file))
        imsave(os.path.join(out_dir, 'labels', '{}.png'.format(basename)),
               imread(os.path.abspath(os.path.join(file, '..', '..', 'labels', '{}.png'.format(basename)))))

    # Class file
    classes = np.stack([(0, 0, 0), (0, 255, 0)])
    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', required=True, type=str, help='Input_folder where the images are')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='Output directory for generated dataset')
    args = vars(parser.parse_args())

    filenames = glob(os.path.join(args.get('input_folder'), '*'))

    # Split 0.2 test, 0.7 train, 0.1 eval
    split, test_split = train_test_split(filenames, test_size=0.2, train_size=0.8, random_state=1)
    train_split, validation_split = train_test_split(split, test_size=0.125, train_size=0.875, random_state=1)

    generate_set(train_split, args.get('output_dir'), 'train')
    generate_set(test_split, args.get('output_dir'), 'test')
    generate_set(validation_split, args.get('output_dir'), 'validation')



