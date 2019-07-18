import os
import shutil
from glob import glob
import cv2
import numpy as np
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
                      size: int,
                      draw_baselines: bool=True,
                      draw_lines: bool=False,
                      draw_endpoints: bool=False,
                      line_thickness: int=10,
                      diameter_endpoint: int=20) -> None:

    page_filename = get_page_filename(image_filename)
    page = PAGE.parse_file(page_filename)
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, pilmode='RGB')
    gt = np.zeros_like(img)

    if draw_baselines:
        gt_baselines = np.zeros_like(img[:, :, 0])
        gt_baselines = cv2.polylines(gt_baselines,
                                     [PAGE.Point.list_to_cv2poly(tl.baseline) for tl in
                                      text_lines],
                                     isClosed=False, color=255,
                                     thickness=int(line_thickness * (gt_baselines.shape[0] / TARGET_HEIGHT)))
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
    save_and_resize(img, os.path.join(output_dir, 'images', '{}.jpg'.format(image_label_basename)), size=size)
    save_and_resize(gt, os.path.join(output_dir, 'labels', '{}.png'.format(image_label_basename)),
                    size=size, nearest=True)
    shutil.copy(page_filename, os.path.join(output_dir, 'gt', '{}.xml'.format(image_label_basename)))


def cbad_set_generator(input_dir: str,
                       output_dir: str,
                       img_size: int,
                       draw_baselines: bool=True,
                       draw_lines: bool=False,
                       line_thickness: int=4,
                       draw_endpoints: bool=False,
                       circle_thickness: int =20):
    """

    :param input_dir: Input directory containing images and PAGE files
    :param output_dir: Output directory to save images and labels
    :param img_size: Size of the resized image (# pixels)
    :param draw_baselines: Draws the baselines (boolean)
    :param draw_lines: Draws the polygon's lines (boolean)
    :param line_thickness: Thickness of annotated baseline
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
    for image_filename in tqdm(image_filenames_list):
        annotate_one_page(image_filename, output_dir, img_size, draw_baselines=draw_baselines,
                          draw_lines=draw_lines, line_thickness=line_thickness,
                          draw_endpoints=draw_endpoints, diameter_endpoint=circle_thickness)

    classes = [(0, 0, 0)]
    if draw_baselines:
        classes.append(DRAWING_COLOR_BASELINES)
    if draw_lines:
        classes.append(DRAWING_COLOR_LINES)
    if draw_endpoints:
        classes.append(DRAWING_COLOR_POINTS)

    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')


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