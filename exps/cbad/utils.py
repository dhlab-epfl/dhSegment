import os
import shutil
import zipfile
import urllib
from glob import glob
import cv2
import numpy as np
import csv
import pandas as pd
from typing import Tuple
from imageio import imread, imsave
from tqdm import tqdm
from dh_segment.io import PAGE

TARGET_HEIGHT = 1100
DRAWING_COLOR_BASELINES = (255, 0, 0)
DRAWING_COLOR_LINES = (0, 255, 0)
DRAWING_COLOR_POINTS = (0, 0, 255)

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def get_page_filename(image_filename: str) -> str:
    page_filename = os.path.join(os.path.dirname(image_filename),
                                 'page',
                                 '{}.xml'.format(os.path.basename(image_filename)[:-4]))

    if os.path.exists(page_filename):
        return page_filename
    else:
        raise FileNotFoundError


def get_image_label_basename(image_filename: str) -> str:
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('_')[0]
    return '{}_{}'.format(acronym, basename.split('.')[0])


def save_and_resize(img: np.array, filename: str, size=None, nearest: bool=False) -> None:
    if size is not None:
        h, w = img.shape[:2]
        ratio = float(np.sqrt(size/(h*w)))
        resized = cv2.resize(img, (int(w*ratio), int(h*ratio)),
                             interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
        imsave(filename, resized)
    else:
        imsave(filename, img)


def annotate_one_page(image_filename: str,
                      output_dir: str,
                      size: int=None,
                      draw_baselines: bool=True,
                      draw_lines: bool=False,
                      draw_endpoints: bool=False,
                      baseline_thickness: float=0.2,
                      diameter_endpoint: int=20) -> Tuple[str, str]:
    """

    :param image_filename:
    :param output_dir:
    :param size: Size of the resized image (# pixels)
    :param draw_baselines: Draws the baselines (boolean)
    :param draw_lines: Draws the polygon's lines (boolean)
    :param draw_endpoints: Predict beginning and end of baselines (True, False)
    :param baseline_thickness: Thickness of annotated baseline (percentage of the line's height)
    :param diameter_endpoint: Diameter of annotated start/end points
    :return:
    """

    page_filename = get_page_filename(image_filename)
    page = PAGE.parse_file(page_filename)
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, pilmode='RGB')
    gt = np.zeros_like(img)

    if text_lines:
        if draw_baselines:
            # Thickness : should be a percentage of the line height, for example 0.2
            mean_line_height, _, _ = _compute_statistics_line_height(page)
            absolute_baseline_thickness = int(max(gt.shape[0]*0.002, baseline_thickness*mean_line_height))

            gt_baselines = np.zeros_like(img[:, :, 0])
            gt_baselines = cv2.polylines(gt_baselines,
                                         [PAGE.Point.list_to_cv2poly(tl.baseline) for tl in
                                          text_lines],
                                         isClosed=False, color=255,
                                         thickness=absolute_baseline_thickness)
            gt[:, :, np.argmax(DRAWING_COLOR_BASELINES)] = gt_baselines

        if draw_lines:
            gt_lines = np.zeros_like(img[:, :, 0])
            for tl in text_lines:
                gt_lines = cv2.fillPoly(gt_lines,
                                        [PAGE.Point.list_to_cv2poly(tl.coords)],
                                        color=255)
            gt[:, :, np.argmax(DRAWING_COLOR_LINES)] = gt_lines

        if draw_endpoints:
            gt_points = np.zeros_like(img[:, :, 0])
            for tl in text_lines:
                try:
                    gt_points = cv2.circle(gt_points, (tl.baseline[0].x, tl.baseline[0].y),
                                           radius=int((diameter_endpoint / 2 * (gt_points.shape[0] / TARGET_HEIGHT))),
                                           color=255, thickness=-1)
                    gt_points = cv2.circle(gt_points, (tl.baseline[-1].x, tl.baseline[-1].y),
                                           radius=int((diameter_endpoint / 2 * (gt_points.shape[0] / TARGET_HEIGHT))),
                                           color=255, thickness=-1)
                except IndexError:
                    print('Length of baseline is {}'.format(len(tl.baseline)))
            gt[:, :, np.argmax(DRAWING_COLOR_POINTS)] = gt_points

    image_label_basename = get_image_label_basename(image_filename)
    output_image_path = os.path.join(output_dir, 'images', '{}.jpg'.format(image_label_basename))
    output_label_path = os.path.join(output_dir, 'labels', '{}.png'.format(image_label_basename))
    save_and_resize(img, output_image_path, size=size)
    save_and_resize(gt, output_label_path, size=size, nearest=True)
    shutil.copy(page_filename, os.path.join(output_dir, 'gt', '{}.xml'.format(image_label_basename)))

    return os.path.abspath(output_image_path), os.path.abspath(output_label_path)


def cbad_set_generator(input_dir: str,
                       output_dir: str,
                       img_size: int,
                       draw_baselines: bool=True,
                       draw_lines: bool=False,
                       line_thickness: float=0.2,
                       draw_endpoints: bool=False,
                       circle_thickness: int =20) -> None:
    """

    :param input_dir: Input directory containing images and PAGE files
    :param output_dir: Output directory to save images and labels
    :param img_size: Size of the resized image (# pixels)
    :param draw_baselines: Draws the baselines (boolean)
    :param draw_lines: Draws the polygon's lines (boolean)
    :param line_thickness: Thickness of annotated baseline (percentage of the line's height)
    :param draw_endpoints: Predict beginning and end of baselines (True, False)
    :param circle_thickness: Diameter of annotated start/end points
    :return:
    """

    # Get image filenames to process
    image_filenames_list = glob('{}/**/*.jpg'.format(input_dir))

    # set
    os.makedirs(os.path.join('{}'.format(output_dir), 'images'))
    os.makedirs(os.path.join('{}'.format(output_dir), 'labels'))
    os.makedirs(os.path.join('{}'.format(output_dir), 'gt'))

    tuples_images_labels = list()
    for image_filename in tqdm(image_filenames_list):
        output_image_path, output_label_path = annotate_one_page(image_filename,
                                                                 output_dir, img_size, draw_baselines=draw_baselines,
                                                                 draw_lines=draw_lines,
                                                                 baseline_thickness=line_thickness,
                                                                 draw_endpoints=draw_endpoints,
                                                                 diameter_endpoint=circle_thickness)

        tuples_images_labels.append((output_image_path, output_label_path))

    # Create classes.txt file
    classes = [(0, 0, 0)]
    if draw_baselines:
        classes.append(DRAWING_COLOR_BASELINES)
    if draw_lines:
        classes.append(DRAWING_COLOR_LINES)
    if draw_endpoints:
        classes.append(DRAWING_COLOR_POINTS)
    if draw_baselines and draw_lines:
        classes.append(tuple(np.array(DRAWING_COLOR_BASELINES) + np.array(DRAWING_COLOR_LINES)))
    if draw_baselines and draw_endpoints:
        classes.append(tuple(np.array(DRAWING_COLOR_BASELINES) + np.array(DRAWING_COLOR_POINTS)))
    if draw_lines and draw_endpoints:
        classes.append(tuple(np.array(DRAWING_COLOR_LINES) + np.array(DRAWING_COLOR_POINTS)))
    if draw_baselines and draw_lines and draw_endpoints:
        classes.append(tuple(np.array(DRAWING_COLOR_BASELINES) + np.array(DRAWING_COLOR_LINES) + np.array(DRAWING_COLOR_POINTS)))

    # Deal with multiclassification
    multiclass_codes = np.greater(classes, len(classes) * [[0, 0, 0]]).astype(int)
    final_classes = np.hstack((classes, multiclass_codes))

    np.savetxt(os.path.join(output_dir, 'classes.txt'), final_classes, fmt='%d')

    with open(os.path.join(output_dir, 'set_data.csv'), 'w') as f:
        writer = csv.writer(f)
        for row in tuples_images_labels:
            writer.writerow(row)


def split_set_for_eval(csv_filename: str) -> None:

    df_data = pd.read_csv(csv_filename, header=None)

    # take 15% for eval
    df_eval = df_data.sample(frac=0.15, random_state=42)
    indexes = df_data.index.difference(df_eval.index)
    df_train = df_data.loc[indexes]

    # save CSVs
    saving_dir = os.path.dirname(csv_filename)
    df_eval.to_csv(os.path.join(saving_dir, 'eval_data.csv'), header=False, index=False, encoding='utf8')
    df_train.to_csv(os.path.join(saving_dir, 'train_data.csv'), header=False, index=False, encoding='utf8')


def draw_lines_fn(xml_filename: str, output_dir: str):
    """
    GIven an XML PAGE file, draws the corresponding lines in the original image.
    :param xml_filename:
    :param output_dir:
    :return:
    """
    basename = os.path.basename(xml_filename).split('.')[0]
    generated_page = PAGE.parse_file(xml_filename)
    drawing_img = generated_page.image_filename
    generated_page.draw_baselines(drawing_img, color=(0, 0, 255))
    imsave(os.path.join(output_dir, '{}.jpg'.format(basename)), drawing_img)


def _compute_statistics_line_height(page_class: PAGE.Page, verbose: bool=False) -> Tuple[float, float, float]:
    """
    Function to compute mean and std of line height among a page.

    :param page_class: json Page
    :param verbose: either to print computational info or not
    :return: tuple (mean, standard deviation, median)
    """
    y_lines_coords = [[c.y for c in tl.coords] for tr in page_class.text_regions for tl in tr.text_lines if tl.coords]
    line_heights = np.array([np.max(y_line_coord) - np.min(y_line_coord) for y_line_coord in y_lines_coords])

    # Remove outliers
    if len(line_heights) > 3:
        outliers = _is_outlier(np.array(line_heights))
        line_heights_filtered = line_heights[~outliers]
    else:
        line_heights_filtered = line_heights
    if verbose:
        print('Considering {}/{} lines to compute line height statistics'.format(len(line_heights_filtered),
                                                                                 len(line_heights)))

    # Compute mean, std, median
    mean = np.mean(line_heights_filtered)
    median = np.median(line_heights_filtered)
    standard_deviation = np.std(line_heights_filtered)

    return mean, standard_deviation, median


def _is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise. Used to find outliers in 1D data.
    https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data

    References:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.

    :param points : An numobservations by numdimensions array of observations
    :param thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    :return: mask : A num_observations-length boolean array.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    # Replace zero values by epsilon
    if not isinstance(med_abs_deviation, float):
        med_abs_deviation = np.maximum(med_abs_deviation, len(med_abs_deviation)*[1e-10])
    else:
        med_abs_deviation = np.maximum(med_abs_deviation, 1e-10)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

# --------


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
              os.path.join(output_dir, 'cbad-icdar2017-test-complex-documents'))
    os.rename(os.path.join(output_dir, 'Test-Baseline Competition - Simple Documents'),
              os.path.join(output_dir, 'cbad-icdar2017-test-simple-documents'))
    os.rename(os.path.join(output_dir, 'Train-Baseline Competition - Complex Documents'),
              os.path.join(output_dir, 'cbad-icdar2017-train-complex-documents'))
    os.rename(os.path.join(output_dir, 'Train-Baseline Competition - Simple Documents'),
              os.path.join(output_dir, 'cbad-icdar2017-train-simple-documents'))

    os.remove(zip_filename)
    print('Files extracted and renamed in {}'.format(output_dir))
