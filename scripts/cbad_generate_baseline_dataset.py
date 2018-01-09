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
P_THICKNESS = 3e-3

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


def process_one(image_filename, output_dir, endpoints=False):
    page = PAGE.parse_file(get_page_filename(image_filename))
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, mode='RGB')
    gt = np.zeros_like(img)
    gt = cv2.polylines(gt,
                       [(PAGE.Point.list_to_cv2poly(tl.baseline)[:, 0, :])[:, None, :] for tl in text_lines],
                       isClosed=False, color=DRAWING_COLOR_BASELINES,
                       thickness=int(P_THICKNESS * gt.shape[0]))

    # Mark end and beginning of baselines
    if endpoints:
        for tl in text_lines:
            gt = cv2.circle(gt, (tl.baseline[0].x, tl.baseline[0].y), radius=int((P_THICKNESS * gt.shape[0]) / 2) + 1,
                            color=DRAWING_COLOR_POINTS, thickness=-1)
            gt = cv2.circle(gt, (tl.baseline[-1].x, tl.baseline[-1].y), radius=int((P_THICKNESS * gt.shape[0]) / 2) + 1,
                            color=DRAWING_COLOR_POINTS, thickness=-1)

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
        process_one(image_filename, '{}/train'.format(args.get('output_dir')), args.get('endpoints'))

    # Validation set
    os.makedirs('{}/validation/images'.format(args.get('output_dir')))
    os.makedirs('{}/validation/labels'.format(args.get('output_dir')))
    for image_filename in tqdm(image_filenames_eval):
        process_one(image_filename, '{}/validation'.format(args.get('output_dir')), args.get('endpoints'))

    # Classes file
    classes = np.stack([(0, 0, 0), DRAWING_COLOR_BASELINES])
    if args.get('endpoints'):
        classes = np.vstack((classes, DRAWING_COLOR_POINTS))
    np.savetxt(os.path.join(args.get('output_dir'), 'classes.txt'), classes, fmt='%d')
