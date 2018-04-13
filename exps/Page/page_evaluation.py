#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
import cv2
import numpy as np
from scipy.misc import imread, imsave
from tqdm import tqdm
from dh_segment.post_processing.boxes_detection import find_boxes
from exps.evaluation.base import Metrics, format_quad_to_string, compare_bin_prediction_to_label, intersection_over_union


def page_evaluate_folder(output_folder: str, validation_dir: str, pixel_wise: bool=True,
                         debug_folder: str=None, verbose: bool=False) -> dict:
    """

    :param output_folder: contains the *.png files from the post_processing
    :param validation_dir: Directory contianing the gt label images
    :param pixel_wise: if True computes pixel-wise accuracy, if False computes IOU accuracy
    :param debug_folder:
    :param verbose:
    :return:
    """
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    filenames_binary_masks = glob(os.path.join(output_folder, '*.png'))

    global_metrics = Metrics()
    list_boxes = list()
    for filename in tqdm(filenames_binary_masks, desc='Evaluation'):
        basename = os.path.basename(filename).split('.')[0]

        # Open post_processed and label image
        post_processed_img = imread(filename)
        post_processed_img = post_processed_img / np.maximum(np.max(post_processed_img), 1)

        label_image = imread(os.path.join(validation_dir, 'labels', '{}.png'.format(basename)), mode='L')
        label_image = label_image / np.max(label_image)

        # Upsample processed image to compare it to original image
        target_shape = (label_image.shape[1], label_image.shape[0])
        bin_upscaled = cv2.resize(np.uint8(post_processed_img), target_shape, interpolation=cv2.INTER_NEAREST)

        if pixel_wise:
            metric = compare_bin_prediction_to_label(bin_upscaled, label_image)
            global_metrics += metric

        pred_box = find_boxes(np.uint8(bin_upscaled), mode='quadrilateral')
        label_box = find_boxes(np.uint8(label_image), mode='quadrilateral', min_area=0.0)

        if debug_folder is not None:
            imsave(os.path.join(debug_folder, '{}_bin.png'.format(basename)), np.uint8(bin_upscaled*255))
            orig_img = imread(os.path.join(validation_dir, 'images', '{}.jpg'.format(basename)))
            if label_box is not None:
                cv2.polylines(orig_img, [label_box[:, None, :]], True, (0, 255, 0), thickness=15)
            else:
                print('There is no labelled page in {}'.format(basename))
            if pred_box is not None:
                cv2.polylines(orig_img, [pred_box[:, None, :]], True, (0, 0, 255), thickness=15)
            else:
                print('No box found in {}'.format(basename))
            imsave(os.path.join(debug_folder, '{}_boxes.jpg'.format(basename)), orig_img)

            list_boxes.append((basename, pred_box))

        if pred_box is not None and label_box is not None:
            iou = intersection_over_union(label_box[:, None, :], pred_box[:, None, :], label_image.shape)
            global_metrics.IOU_list.append(iou)
        else:
            global_metrics.IOU_list.append(0)
            if verbose:
                print('No box found for {}'.format(basename))

    if debug_folder:
        with open(os.path.join(debug_folder, 'predicted_boxes.txt'), 'w') as f:
            for b in list_boxes:
                s = '{},{}\n'.format(b[0], format_quad_to_string(b))
                f.write(s)

    if pixel_wise:
        global_metrics.compute_prf()

        print('EVAL --- R : {}, P : {}, FM : {}\n'.format(global_metrics.recall, global_metrics.precision,
                                                          global_metrics.f_measure))

    global_metrics.compute_miou()
    print('EVAL --- mIOU : {}\n'.format(global_metrics.mIOU))
    # Export txt similar to test txt ?

    return {
        'precision': global_metrics.precision,
        'recall': global_metrics.recall,
        'f_measure': global_metrics.f_measure,
        'mIOU': global_metrics.mIOU
    }