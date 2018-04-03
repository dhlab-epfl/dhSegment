import os
from glob import glob
from scipy.misc import imread, imsave, imresize
import cv2
import numpy as np
from cini_post_processing import cini_post_processing_fn
from doc_seg.utils import load_pickle
import pandas as pd


def cini_evaluate_folder(output_folder: str, validation_dir: str, verbose=False, debug_folder=None) -> dict:
    filenames_processed = glob(os.path.join(output_folder, '*.pkl'))

    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    iou_cardboards = []
    iou_images = []
    basenames = []
    for filename in filenames_processed:
        basename = os.path.basename(filename).split('.')[0]

        data = load_pickle(filename)

        cardboard_coords, image_coords, shape = data['cardboard_rectangle'], data['image_rectangle'], data['shape']

        # Open label image
        label_path = os.path.join(validation_dir, 'labels', '{}.png'.format(basename))
        if not os.path.exists(label_path):
            label_path = label_path.replace('.png', '.jpg')
        label_image = imread(label_path, mode='RGB')
        label_image = imresize(label_image, shape[:2])
        label_predictions = np.stack([
            label_image[:, :, 0] > 250,
            label_image[:, :, 1] > 250,
            label_image[:, :, 2] > 250
        ], axis=-1).astype(np.float32)
        label_cardboard_coords, label_image_coords = cini_post_processing_fn(label_predictions,
                                                                             clean_predictions=False)

        # Compute errors
        def intersection_over_union(cnt1, cnt2):
            mask1 = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(mask1, cv2.boxPoints(cnt1).astype(np.int32), 1)
            mask2 = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(mask2, cv2.boxPoints(cnt2).astype(np.int32), 1)
            return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        iou_cardboard = intersection_over_union(label_cardboard_coords, cardboard_coords)
        iou_image = intersection_over_union(label_image_coords, image_coords)

        iou_cardboards.append(iou_cardboard)
        iou_images.append(iou_image)
        basenames.append(basename)

        if debug_folder is not None:
            img_filename = os.path.join(validation_dir, 'images', '{}.jpg'.format(basename))
            img = imresize(imread(img_filename), shape)
            cv2.polylines(img, cv2.boxPoints(cardboard_coords).astype(np.int32)[None], True, (255, 0, 0), 4)
            cv2.polylines(img, cv2.boxPoints(image_coords).astype(np.int32)[None], True, (0, 0, 255), 4)
            imsave(os.path.join(debug_folder, '{}.jpg'.format(basename)), img)

    result = {
        'cardboard_mean_iou': np.mean(iou_cardboards),
        'image_mean_iou': np.mean(iou_images),
    }

    if debug_folder is not None:
        df = pd.DataFrame(data=list(zip(basenames, iou_cardboards, iou_images)),
                          columns=['basename', 'iou_cardboard', 'iou_image'])
        df = df.sort_values('iou_image', ascending=True)
        df.to_csv(os.path.join(debug_folder, 'scores.csv'), index=False)

    if verbose:
        print(result)
    return result


