import os
from doc_seg_datasets import PAGE
import cv2
from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm


TARGET_WIDTH = 1200
INPUT_DIR = '/home/seguin/document_datasets/layout_analysis'
OUTPUT_DIR = '/scratch/benoit/layout_analysis_1200'


def save_and_resize(img, filename, nearest=False):
    resized = cv2.resize(img, (TARGET_WIDTH, (img.shape[0]*TARGET_WIDTH)//img.shape[1]),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    imsave(filename, resized)


def process_one(image_dir, page_dir, output_dir, basename, colormap, color_labels):
    image_filename = os.path.join(image_dir, "{}.jpg".format(basename))
    page_filename = os.path.join(page_dir, "{}.xml".format(basename))

    page = PAGE.parse_file(page_filename)
    text_lines = [tl for tr in page.text_regions for tl in tr.text_lines]
    graphic_regions = page.graphic_regions
    img = imread(image_filename, mode='RGB')

    gt = np.zeros_like(img[:, :, 0])
    mask1 = cv2.fillPoly(gt.copy(), [PAGE.Point.list_to_cv2poly(tl.coords)
                                     for tl in text_lines if 'comment' in tl.id], 1)
    mask2 = cv2.fillPoly(gt.copy(), [PAGE.Point.list_to_cv2poly(tl.coords)
                                     for tl in text_lines if not 'comment' in tl.id], 1)
    mask3 = cv2.fillPoly(gt.copy(), [PAGE.Point.list_to_cv2poly(tl.coords)
                                     for tl in graphic_regions], 1)
    arr = np.dstack([mask1, mask2, mask3])

    gt_img = convert_array_masks(arr, colormap, color_labels)
    save_and_resize(img, os.path.join(output_dir, 'images', '{}.jpg'.format(basename)))
    save_and_resize(gt_img, os.path.join(output_dir, 'labels', '{}.png'.format(basename)), nearest=True)


def make_cmap(N_CLASSES):
    # Generate the colors for the classes (with background class being 0,0,0)
    c_size = 2**N_CLASSES - 1
    cmap = np.concatenate([[[0, 0, 0]], plt.cm.Set1(np.arange(c_size) / (c_size))[:, :3]])
    cmap = (cmap * 255).astype(np.uint8)
    assert N_CLASSES <= 8, "ARGH!! can not handle more than 8 classes"
    c_full_label = np.unpackbits(np.arange(2 ** N_CLASSES).astype(np.uint8)[:, None], axis=-1)[:, -N_CLASSES:]
    return cmap, c_full_label


def convert_array_masks(arr, cmap, c_full_label):
    N_CLASSES = arr.shape[-1]
    c = np.zeros((2,) * N_CLASSES, np.int32)
    for i, inds in enumerate(c_full_label):
        c[tuple(inds)] = i
    c_ind = c[[arr[:, :, i] for i in range(arr.shape[-1])]]
    return cmap[c_ind]


def save_cmap_to_txt(filename, cmap, c_full_label):
    np.savetxt(filename, np.concatenate([cmap, c_full_label], axis=1), fmt='%i')


colormap, color_labels = make_cmap(3)

train_basenames = [os.path.basename(p)[:-4] for p in glob('{}/img/training/*.jpg'.format(INPUT_DIR))]
os.makedirs('{}/train/images'.format(OUTPUT_DIR))
os.makedirs('{}/train/labels'.format(OUTPUT_DIR))
for basename in tqdm(train_basenames):
    process_one(os.path.join(INPUT_DIR, 'img', 'training'),
                os.path.join(INPUT_DIR, 'PAGE-gt', 'training'),
                os.path.join(OUTPUT_DIR, 'train'),
                basename, colormap, color_labels)


val_basenames = [os.path.basename(p)[:-4] for p in glob('{}/img/validation/*.jpg'.format(INPUT_DIR))]
os.makedirs('{}/eval/images'.format(OUTPUT_DIR))
os.makedirs('{}/eval/labels'.format(OUTPUT_DIR))
for basename in tqdm(val_basenames):
    process_one(os.path.join(INPUT_DIR, 'img', 'validation'),
                os.path.join(INPUT_DIR, 'PAGE-gt', 'validation'),
                os.path.join(OUTPUT_DIR, 'eval'),
                basename, colormap, color_labels)


test_basenames = [os.path.basename(p)[:-4] for p in glob('{}/img/public-test/*.jpg'.format(INPUT_DIR))]
os.makedirs('{}/test/images'.format(OUTPUT_DIR))
os.makedirs('{}/test/labels'.format(OUTPUT_DIR))
for basename in tqdm(test_basenames):
    process_one(os.path.join(INPUT_DIR, 'img', 'public-test'),
                os.path.join(INPUT_DIR, 'PAGE-gt', 'public-test'),
                os.path.join(OUTPUT_DIR, 'test'),
                basename, colormap, color_labels)

save_cmap_to_txt(os.path.join(OUTPUT_DIR, 'train', 'classes.txt'), colormap, color_labels)