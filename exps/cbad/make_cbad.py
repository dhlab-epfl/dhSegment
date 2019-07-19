#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
import click
from utils import cbad_download, cbad_set_generator

TRAIN_COMPLEX_DIR = 'cbad-icdar2017-train-complex-documents'
TRAIN_SIMPLE_DIR = 'cbad-icdar2017-train-simple-documents'
TEST_COMPLEX_DIR = 'cbad-icdar2017-test-complex-documents'
TEST_SIMPLE_DIR = 'cbad-icdar2017-test-simple-documents'


@click.command()
@click.option('--downloading_dir', help='Directory to download the cBAD-ICDAR17 dataset')
@click.option('--masks_dir', help="Directory where to output the generated masks")
def generate_cbad_dataset(downloading_dir: str, masks_dir: str):
    # Check if dataset has already been downloaded
    if os.path.exists(downloading_dir):
        print('Dataset has already been downloaded. Skipping process.')
    else:
        # Download dataset
        cbad_download(downloading_dir)

    # Create masks
    dirs_tuple = [(os.path.join(downloading_dir, TRAIN_COMPLEX_DIR), os.path.join(masks_dir, 'complex', 'train')),
                  (os.path.join(downloading_dir, TEST_COMPLEX_DIR), os.path.join(masks_dir, 'complex', 'test')),
                  (os.path.join(downloading_dir, TRAIN_SIMPLE_DIR), os.path.join(masks_dir, 'simple', 'train')),
                  (os.path.join(downloading_dir, TEST_SIMPLE_DIR), os.path.join(masks_dir, 'simple', 'test'))]

    print('Creating sets')
    for dir_tuple in dirs_tuple:
        input_dir, output_dir = dir_tuple
        os.makedirs(output_dir, exist_ok=True)
        cbad_set_generator(input_dir=input_dir,
                           output_dir=output_dir,
                           img_size=2e6,
                           draw_baselines=True,
                           draw_lines=True,
                           line_thickness=5,
                           draw_endpoints=True,
                           circle_thickness=15)


if __name__ == '__main__':
    generate_cbad_dataset()
