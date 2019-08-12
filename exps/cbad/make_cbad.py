#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
import click
# from utils import cbad_download, cbad_set_generator, split_set_for_eval
from utils import cbad_set_generator, split_set_for_eval
from exps.commonutils import cbad_download, CBAD_TRAIN_COMPLEX_FOLDER, CBAD_TEST_COMPLEX_FOLDER, CBAD_TRAIN_SIMPLE_FOLDER, CBAD_TEST_SIMPLE_FOLDER


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
    dirs_tuple = [(os.path.join(downloading_dir, CBAD_TRAIN_COMPLEX_FOLDER), os.path.join(masks_dir, 'complex', 'train')),
                  (os.path.join(downloading_dir, CBAD_TEST_COMPLEX_FOLDER), os.path.join(masks_dir, 'complex', 'test')),
                  (os.path.join(downloading_dir, CBAD_TRAIN_SIMPLE_FOLDER), os.path.join(masks_dir, 'simple', 'train')),
                  (os.path.join(downloading_dir, CBAD_TEST_SIMPLE_FOLDER), os.path.join(masks_dir, 'simple', 'test'))]

    print('Creating sets')
    for dir_tuple in dirs_tuple:
        input_dir, output_dir = dir_tuple
        os.makedirs(output_dir, exist_ok=True)
        # For each set create the folder with the annotated data
        cbad_set_generator(input_dir=input_dir,
                           output_dir=output_dir,
                           img_size=2e6,
                           draw_baselines=True,
                           draw_endpoints=False)

        # Split the 'official' train set into training and validation set
        if 'train' in output_dir:
            print('Make eval set from the given training data (0.15/0.85 eval/train)')
            csv_filename = os.path.join(output_dir, 'set_data.csv')
            split_set_for_eval(csv_filename)
            print('Done!')


if __name__ == '__main__':
    generate_cbad_dataset()
