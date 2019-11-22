#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
from tqdm import tqdm
import urllib
import zipfile
import numpy as np
import cv2
from imageio import imsave

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

CBAD_TRAIN_COMPLEX_FOLDER = 'cbad-icdar2017-train-complex-documents'
CBAD_TEST_COMPLEX_FOLDER = 'cbad-icdar2017-test-complex-documents'
CBAD_TRAIN_SIMPLE_FOLDER = 'cbad-icdar2017-train-simple-documents'
CBAD_TEST_SIMPLE_FOLDER = 'cbad-icdar2017-test-simple-documents'


def get_page_filename(image_filename: str) -> str:
    """
    Given an path to a .jpg or .png file, get the corresponding .xml file.

    :param image_filename: filename of the image
    :return: the filename of the corresponding .xml file, raises exception if .xml file does not exist
    """
    page_filename = os.path.join(os.path.dirname(image_filename),
                                 'page',
                                 '{}.xml'.format(os.path.basename(image_filename)[:-4]))

    if os.path.exists(page_filename):
        return page_filename
    else:
        raise FileNotFoundError


def get_image_label_basename(image_filename: str) -> str:
    """
    Creates a new filename composed of the begining of the folder/collection (ex. EPFL, ABP) and the original filename

    :param image_filename: path of the image filename
    :return:
    """
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('_')[0]
    return '{}_{}'.format(acronym, basename.split('.')[0])


def save_and_resize(img: np.array,
                    filename: str,
                    size=None,
                    nearest: bool=False) -> None:
    """
    Resizes the image if necessary and saves it. The resizing will keep the image ratio

    :param img: the image to resize and save (numpy array)
    :param filename: filename of the saved image
    :param size: size of the image after resizing (in pixels). The ratio of the original image will be kept
    :param nearest: whether to use nearest interpolation method (default to False)
    :return:
    """
    if size is not None:
        h, w = img.shape[:2]
        ratio = float(np.sqrt(size/(h*w)))
        resized = cv2.resize(img, (int(w*ratio), int(h*ratio)),
                             interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        imsave(filename, resized)
    else:
        imsave(filename, img)


# ------------------------------


def _progress_hook(t):
    last_b = [0]

    def update_to(b: int=1, bsize: int=1, tsize: int=None):
        """
        Adapted from: source unknown
        :param b: Number of blocks transferred so far [default: 1].
        :param bsize: Size of each block (in tqdm units) [default: 1].
        :param tsize: Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def cbad_download(output_dir: str):
    """
    Download BAD-READ dataset.

    :param output_dir: folder where to download the data
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_filename = os.path.join(output_dir, 'cbad-icdar17.zip')

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading cBAD-ICDAR17 dataset") as t:
        urllib.request.urlretrieve('https://zenodo.org/record/1491441/files/READ-ICDAR2017-cBAD-dataset-v4.zip',
                                   zip_filename, reporthook=_progress_hook(t))
    print('cBAD-ICDAR2017 dataset downloaded successfully!')
    print('Extracting files ...')
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Renaming
    os.rename(os.path.join(output_dir, 'Test-Baseline Competition - Complex Documents'),
              os.path.join(output_dir, CBAD_TEST_COMPLEX_FOLDER))
    os.rename(os.path.join(output_dir, 'Test-Baseline Competition - Simple Documents'),
              os.path.join(output_dir, CBAD_TEST_SIMPLE_FOLDER))
    os.rename(os.path.join(output_dir, 'Train-Baseline Competition - Complex Documents'),
              os.path.join(output_dir, CBAD_TRAIN_COMPLEX_FOLDER))
    os.rename(os.path.join(output_dir, 'Train-Baseline Competition - Simple Documents'),
              os.path.join(output_dir, CBAD_TRAIN_SIMPLE_FOLDER))

    os.remove(zip_filename)
    print('Files extracted and renamed in {}'.format(output_dir))
