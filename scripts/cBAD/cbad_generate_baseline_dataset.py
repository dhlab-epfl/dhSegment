from doc_seg_datasets import PAGE
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from scipy.misc import imread, imsave
import argparse


TARGET_HEIGHT = 1100
DRAWING_COLOR_BASELINES = (255, 0, 0)
DRAWING_COLOR_POINTS = (0, 255, 0)
# LINE_THICKNESS = 10  # 3e-3
# DIAMETER_ENDPOINT = 20

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def get_page_filename(image_filename):
    return os.path.join(os.path.dirname(image_filename),
                        'page',
                        '{}.xml'.format(os.path.basename(image_filename)[:-4]))


def get_image_label_basename(image_filename):
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    acronym = directory.split(os.path.sep)[-1].split('_')[0]
    return '{}_{}'.format(acronym, basename.split('.')[0])


def save_and_resize(img, filename, nearest=False):
    resized = cv2.resize(img, ((img.shape[1]*TARGET_HEIGHT)//img.shape[0], TARGET_HEIGHT),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    imsave(filename, resized)


def process_one(image_filename, output_dir, endpoints=False, line_thickness=10, diameter_endpoint=20):
    page = PAGE.parse_file(get_page_filename(image_filename))
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, mode='RGB')
    gt = np.zeros_like(img)

    if not endpoints:
        gt = cv2.polylines(gt,
                           [(PAGE.Point.list_to_cv2poly(tl.baseline)[:, 0, :])[:, None, :] for tl in text_lines],
                           isClosed=False, color=DRAWING_COLOR_BASELINES,
                           thickness=int(line_thickness * (gt.shape[0] / TARGET_HEIGHT)))

    else:
        gt_lines = np.zeros_like(img[:, :, 0])
        gt_lines = cv2.polylines(gt_lines,
                           [(PAGE.Point.list_to_cv2poly(tl.baseline)[:, 0, :])[:, None, :] for tl in text_lines],
                           isClosed=False, color=255,
                           thickness=int(line_thickness * (gt_lines.shape[0] / TARGET_HEIGHT)))

        gt_points = np.zeros_like(img[:, :, 0])
        for tl in text_lines:
            try:
                gt_points = cv2.circle(gt_points, (tl.baseline[0].x, tl.baseline[0].y),
                                       radius=int((diameter_endpoint/2 * (gt_points.shape[0]/TARGET_HEIGHT))),
                                       color=255, thickness=-1)
                gt_points = cv2.circle(gt_points, (tl.baseline[-1].x, tl.baseline[-1].y),
                                       radius=int((diameter_endpoint/2 * (gt_points.shape[0]/TARGET_HEIGHT))),
                                       color=255, thickness=-1)
            except IndexError:
                print('Length of baseline is {}'.format(len(tl.baseline)))

        gt[:, :, np.argmax(DRAWING_COLOR_BASELINES)] = gt_lines
        gt[:, :, np.argmax(DRAWING_COLOR_POINTS)] = gt_points

    save_and_resize(img, os.path.join(output_dir, 'images', '{}.jpg'.format(get_image_label_basename(image_filename))))
    save_and_resize(gt, os.path.join(output_dir, 'labels', '{}.png'.format(get_image_label_basename(image_filename))),
                    nearest=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, type=str, default=None,
                        help='Input directory containing images and PAGE files')
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None,
                        help='Output directory to save images and labels')
    parser.add_argument('-e', '--endpoints', required=False, type=bool, default=False,
                        help='Predict beginning and end of baselines')
    parser.add_argument('-l', '--line_thickness', type=int, default=10, help='Thickness of annotated baseline')
    parser.add_argument('-c', '--circle_thickness', type=int, default=20, help='Diameter of annotated start/end points')
    args = vars(parser.parse_args())

    # Get image filenames to process
    image_filenames = glob('{}/**/*.jpg'.format(args.get('input_dir')))

    # Split data into training and validation set (0.9/0.1)
    train_inds = np.random.choice(len(image_filenames), size=int(0.9 * len(image_filenames)), replace=False)
    train_mask = np.zeros(len(image_filenames), dtype=np.bool_)
    train_mask[train_inds] = 1
    image_filenames_train = np.array(image_filenames)[train_mask]
    image_filenames_eval = np.array(image_filenames)[~train_mask]

    # Train set
    os.makedirs('{}/train/images'.format(args.get('output_dir')))
    os.makedirs('{}/train/labels'.format(args.get('output_dir')))
    for image_filename in tqdm(image_filenames_train):
        process_one(image_filename, '{}/train'.format(args.get('output_dir')), args.get('endpoints'),
                    args.get('line_thickness'), args.get('circle_thickness'))

    # Validation set
    os.makedirs('{}/validation/images'.format(args.get('output_dir')))
    os.makedirs('{}/validation/labels'.format(args.get('output_dir')))
    for image_filename in tqdm(image_filenames_eval):
        process_one(image_filename, '{}/validation'.format(args.get('output_dir')), args.get('endpoints'),
                    args.get('line_thickness'), args.get('circle_thickness'))

    # Classes file
    classes = np.stack([(0, 0, 0), DRAWING_COLOR_BASELINES])
    if args.get('endpoints'):
        classes = np.vstack((classes, DRAWING_COLOR_POINTS))
        classes = np.vstack((classes, np.array(DRAWING_COLOR_BASELINES) + np.array(DRAWING_COLOR_POINTS)))
        # For multilabels, we should add the code for each color after the RGB values (R G B Code)
        codes_list = list()
        n_bits = len('{:b}'.format(classes.shape[0] - 1))
        for i in range(classes.shape[0]):
            codes_list.append('{:08b}'.format(i)[-n_bits:])
        codes_ints = [[int(char) for char in code] for code in codes_list]
        classes = np.hstack((classes, np.array(codes_ints)))

    np.savetxt(os.path.join(args.get('output_dir'), 'classes.txt'), classes, fmt='%d')
