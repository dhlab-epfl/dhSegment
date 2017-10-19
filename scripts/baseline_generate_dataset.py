from doc_seg_datasets import PAGE
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from scipy.misc import imread, imsave

INPUT_DIR = '/home/seguin/basline/Baseline Competition - Simple Documents'
OUTPUT_DIR = '/scratch/benoit/baseline'
TARGET_WIDTH = 600
DRAWING_COLOR = (255, 0, 0)

image_filenames = glob('{}/**/*.jpg'.format(INPUT_DIR))

def get_page_filename(image_filename):
    return os.path.dirname(image_filename)+'/page/{}.xml'.format(os.path.basename(image_filename)[:-4])

def save_and_resize(img, filename):
    resized = cv2.resize(img, (TARGET_WIDTH, (img.shape[0]*TARGET_WIDTH)//img.shape[1]))
    imsave(filename, resized)

def process_one(image_filename, output_dir, basename):
    page = PAGE.parse_file(get_page_filename(image_filename))
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    img = imread(image_filename, mode='RGB')
    gt = np.zeros_like(img)
    cv2.fillPoly(gt, [PAGE.Point.list_to_cv2poly(tl.coords) for tl in text_lines], DRAWING_COLOR)
    save_and_resize(img, os.path.join(output_dir, 'images', '{}.jpg'.format(basename)))
    save_and_resize(gt, os.path.join(output_dir, 'labels', '{}.png'.format(basename)))

    classes = np.stack([(0, 0, 0), DRAWING_COLOR])
    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')


train_inds = np.random.choice(len(image_filenames), size=int(0.8*len(image_filenames)), replace=False)
train_mask = np.zeros(len(image_filenames), dtype=np.bool_)
train_mask[train_inds] = 1
image_filenames_train = np.array(image_filenames)[train_mask]
image_filenames_eval = np.array(image_filenames)[~train_mask]

os.makedirs('{}/train/images'.format(OUTPUT_DIR))
os.makedirs('{}/train/labels'.format(OUTPUT_DIR))
for i, image_filename in enumerate(tqdm(image_filenames_train)):
    process_one(image_filename, '{}/train'.format(OUTPUT_DIR), '{:05d}'.format(i))

os.makedirs('{}/eval/images'.format(OUTPUT_DIR))
os.makedirs('{}/eval/labels'.format(OUTPUT_DIR))
for i, image_filename in enumerate(tqdm(image_filenames_eval)):
    process_one(image_filename, '{}/eval'.format(OUTPUT_DIR), '{:05d}'.format(i))
