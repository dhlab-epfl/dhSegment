#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from imageio import imread, imsave
import numpy as np
import cv2
import os
import re
from tqdm import tqdm
import urllib.request
from exps.commonutils import _progress_hook, CBAD_TEST_SIMPLE_FOLDER, CBAD_TEST_COMPLEX_FOLDER, \
    CBAD_TRAIN_SIMPLE_FOLDER, CBAD_TRAIN_COMPLEX_FOLDER

TRAIN_FILE_URL = 'https://raw.githubusercontent.com/ctensmeyer/pagenet/master/annotations/cbad_train_annotator_1.txt'
TEST_FILE_URL = 'https://raw.githubusercontent.com/ctensmeyer/pagenet/master/annotations/cbad_test_annotator_1.txt'
EVAL_FILE_URL = 'https://raw.githubusercontent.com/ctensmeyer/pagenet/master/annotations/cbad_val_annotator_1.txt'

TRAIN_TXT_FILENAME = 'page_train.txt'
TEST_TXT_FILENAME = 'page_test.txt'
EVAL_TXT_FILENAME = 'page_eval.txt'


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


def page_set_annotator(txt_filename: str, output_dir: str):
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
        coords, image_filename = get_coords_form_txt_line(line)

        try:
            img = imread(image_filename)
        except FileNotFoundError:
            print('File {} not found'.format(image_filename))
            continue
        label_img = np.zeros((img.shape[0], img.shape[1], 3))

        label_img = cv2.fillPoly(label_img, [coords], (255, 0, 0))
        # if coords_double is not None:
        #     label_img = cv2.polylines(label_img, [coords_double], False, color=(0, 0, 0), thickness=50)

        collection, filename = image_filename.split(os.path.sep)[-2:]

        imsave(os.path.join(output_img_dir, '{}_{}.jpg'.format(collection.split('_')[0], filename.split('.')[0])), img.astype(np.uint8))
        imsave(os.path.join(output_label_dir, '{}_{}.png'.format(collection.split('_')[0], filename.split('.')[0])), label_img.astype(np.uint8))

    # Class file
    classes = np.stack([(0, 0, 0), (255, 0, 0)])
    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')

# -----------------------------


def page_files_download(output_dir: str) -> None:
    """
    Download Page txt files from github repository.

    :param output_dir: folder where to download the data
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    train_filename = os.path.join(output_dir, TRAIN_TXT_FILENAME)
    test_filename = os.path.join(output_dir, TEST_TXT_FILENAME)
    eval_filename = os.path.join(output_dir, EVAL_TXT_FILENAME)

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading train file") as t:
        urllib.request.urlretrieve(TRAIN_FILE_URL, train_filename, reporthook=_progress_hook(t))
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading test file") as t:
        urllib.request.urlretrieve(TEST_FILE_URL, test_filename, reporthook=_progress_hook(t))
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading eval file") as t:
        urllib.request.urlretrieve(EVAL_FILE_URL, eval_filename, reporthook=_progress_hook(t))

    print('Page files downloaded successfully!')


def format_txt_file(input_txt_filename: str,
                    output_txt_filename: str,
                    cbad_data_folder: str) -> None:
    """
    Transforms the relative path of the images into absolute path.

    :param input_txt_filename: original downloaded .txt filename
    :param output_txt_filename: filename of the formatted content
    :param cbad_data_folder: path to the folder containing the READ-BAD data
    :return:
    """
    final_tokens = list()
    for line in open(input_txt_filename, 'r'):
        tokens = line.split(',')
        filename = tokens[0]
        full_filename = os.path.join(os.path.abspath(cbad_data_folder), filename)

        if 'complex' in filename:
            pattern = 'complex'
            candidate_folders = [CBAD_TRAIN_COMPLEX_FOLDER, CBAD_TEST_COMPLEX_FOLDER]
        elif 'simple' in filename:
            pattern = 'simple'
            candidate_folders = [CBAD_TRAIN_SIMPLE_FOLDER, CBAD_TEST_SIMPLE_FOLDER]
        else:
            raise Exception

        option1 = re.sub(pattern, candidate_folders[0], full_filename)
        option2 = re.sub(pattern, candidate_folders[1], full_filename)
        # .JPG files
        option3 = re.sub(pattern, candidate_folders[0], full_filename.split('.')[0] + '.JPG')
        option4 = re.sub(pattern, candidate_folders[1], full_filename.split('.')[0] + '.JPG')

        if os.path.exists(option1):
            tokens[0] = option1
        elif os.path.exists(option2):
            tokens[0] = option2
        elif os.path.exists(option3):
            tokens[0] = option3
        elif os.path.exists(option4):
            tokens[0] = option4
        else:
            raise FileNotFoundError('for {}'.format(filename))

        final_tokens.append(','.join(tokens))

    with open(output_txt_filename, 'w') as f:
        for line in final_tokens:
            f.write(line)
