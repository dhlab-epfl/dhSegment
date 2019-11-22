#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
import click
from exps.commonutils import cbad_download
from utils import page_files_download, page_set_annotator, \
    format_txt_file, TRAIN_TXT_FILENAME, TEST_TXT_FILENAME, EVAL_TXT_FILENAME


@click.command()
@click.option('--downloading_dir', help='Directory to download the cBAD-ICDAR17 dataset')
@click.option('--masks_dir', help="Directory where to output the generated masks")
def generate_page_dataset(downloading_dir: str, masks_dir: str):
    # Check if dataset has already been downloaded
    if os.path.exists(downloading_dir):
        print('Dataset has already been downloaded at {}. Skipping process.'.format(downloading_dir))
    else:
        # Download dataset
        cbad_download(downloading_dir)

    page_txt_folder = os.path.join(downloading_dir, 'page-txt-files')
    if os.path.exists(page_txt_folder):
        print('Page txt files have already been downloaded at {}. Skipping process.'.format(page_txt_folder))
    else:
        # Download files
        page_files_download(page_txt_folder)

    tuple_train = (os.path.join(page_txt_folder, TRAIN_TXT_FILENAME),
                   '_formatted.'.join(TRAIN_TXT_FILENAME.split('.')),
                   os.path.join(masks_dir, 'train'))
    tuple_test = (os.path.join(page_txt_folder, TEST_TXT_FILENAME),
                  '_formatted.'.join(TEST_TXT_FILENAME.split('.')),
                  os.path.join(masks_dir, 'test'))
    tuple_eval = (os.path.join(page_txt_folder, EVAL_TXT_FILENAME),
                  '_formatted.'.join(EVAL_TXT_FILENAME.split('.')),
                  os.path.join(masks_dir, 'eval'))

    print('Creating sets')
    for tup in [tuple_train, tuple_test, tuple_eval]:
        input_txt_filename, output_txt_filename, set_masks_dir = tup

        # Format txt files
        format_txt_file(input_txt_filename, output_txt_filename, downloading_dir)

        # Create masks
        os.makedirs(set_masks_dir, exist_ok=True)
        page_set_annotator(output_txt_filename, set_masks_dir)

    print('Done!')

if __name__ == '__main__':
    generate_page_dataset()
