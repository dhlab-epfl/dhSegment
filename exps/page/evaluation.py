#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from tqdm import tqdm
from glob import glob
import os
from imageio import imread
import numpy as np
from .process import extract_page
from dh_segment.utils.evaluation import intersection_over_union, Metrics


PP_PARAMS = {'threshold': -1, 'kernel_size': 5}


def eval_fn(input_dir: str, groundtruth_dir: str, post_process_params: dict=PP_PARAMS) -> Metrics:
    """

    :param input_dir: directory containing the predictions .npy files (range [0, 255])
    :param groundtruth_dir: directory containing the ground truth images (.png) (must have the same name as predictions
                            files in input_dir)
    :param post_process_params: params for post processing fn
    :return: Metrics object containing all the necessary metrics
    """
    global_metrics = Metrics()
    for file in tqdm(glob(os.path.join(input_dir, '*.npy'))):
        basename = os.path.basename(file).split('.')[0]

        prediction = np.load(file)
        label_image = imread(os.path.join(groundtruth_dir, '{}.png'.format(basename)), pilmode='L')

        pred_box = extract_page(prediction / np.max(prediction), **post_process_params)
        label_box = extract_page(label_image / np.max(label_image), min_area=0.0)

        if pred_box is not None and label_box is not None:
            iou = intersection_over_union(label_box[:, None, :], pred_box[:, None, :], label_image.shape)
            global_metrics.IOU_list.append(iou)
        else:
            global_metrics.IOU_list.append(0)

    global_metrics.compute_miou()
    print('EVAL --- mIOU : {}\n'.format(global_metrics.mIOU))

    return global_metrics
