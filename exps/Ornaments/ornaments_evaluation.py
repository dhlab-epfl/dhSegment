#!/usr/bin/env python
__author__ = 'solivr'

from exps.evaluation.base import Metrics
from exps.post_processing.boxes_detection import find_box
import os
from glob import glob
from tqdm import tqdm
from scipy.misc import imread, imsave
import numpy as np
import cv2


def ornament_evaluate_folder(output_folder: str, validation_dir: str, debug_folder: str=None,
                             verbose: bool=False, min_area: float=0.0, miou_threshold: float=0.8) -> dict:

    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    filenames_binary_masks = glob(os.path.join(output_folder, '*.png'))

    global_metrics = Metrics()
    for filename in tqdm(filenames_binary_masks, desc='Evaluation'):
        basename = os.path.basename(filename).split('.')[0]

        # Open post_processed and label image
        post_processed_img = imread(filename)
        post_processed_img = post_processed_img / np.maximum(np.max(post_processed_img), 1)

        label_image = imread(os.path.join(validation_dir, 'labels', '{}.png'.format(basename)), mode='L')
        label_image = label_image / np.max(label_image) if np.max(label_image) > 0 else label_image

        # Upsample processed image to compare it to original image
        target_shape = (label_image.shape[1], label_image.shape[0])
        bin_upscaled = cv2.resize(np.uint8(post_processed_img), target_shape, interpolation=cv2.INTER_NEAREST)

        pred_boxes = find_box(np.uint8(bin_upscaled), mode='min_rectangle', min_area=min_area, n_max_boxes=np.inf)
        label_boxes = find_box(np.uint8(label_image), mode='min_rectangle', min_area=min_area, n_max_boxes=np.inf)

        if debug_folder is not None:
            # imsave(os.path.join(debug_folder, '{}_bin.png'.format(basename)), np.uint8(bin_upscaled*255))
            # orig_img = imread(os.path.join(validation_dir, 'images', '{}.jpg'.format(basename)), mode='RGB')
            orig_img = imread(os.path.join(validation_dir, 'images', '{}.png'.format(basename)), mode='RGB')
            cv2.polylines(orig_img, [lb[:, None, :] for lb in label_boxes], True, (0, 255, 0), thickness=15)
            if pred_boxes is not None:
                cv2.polylines(orig_img, [pb[:, None, :] for pb in pred_boxes], True, (0, 0, 255), thickness=15)
            imsave(os.path.join(debug_folder, '{}_boxes.jpg'.format(basename)), orig_img)

        def intersection_over_union(cnt1, cnt2):
            mask1 = np.zeros_like(label_image)
            mask1 = cv2.fillConvexPoly(mask1, cnt1.astype(np.int32), 1).astype(np.int8)
            mask2 = np.zeros_like(label_image)
            mask2 = cv2.fillConvexPoly(mask2, cnt2.astype(np.int32), 1).astype(np.int8)

            return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        def compute_metric_boxes(predicted_boxes: np.array, label_boxes: np.array, threshold: float=miou_threshold):
            # Todo test this fn
            metric = Metrics()
            if label_boxes is None:
                if predicted_boxes is None:
                    metric.true_negatives += 1
                    metric.total_elements += 1
                else:
                    metric.false_negatives += len(predicted_boxes)

            else:
                for pb in predicted_boxes:
                    best_iou = 0
                    for lb in label_boxes:
                        iou = intersection_over_union(pb[:, None, :], lb[:, None, :])
                        if iou > best_iou:
                            best_iou = iou

                    if best_iou > threshold:
                        metric.true_positives += 1
                        metric.IOU_list.append(best_iou)
                    elif best_iou < 0.1:
                        metric.false_negatives += 1
                    else:
                        metric.false_positives += 1
                        metric.IOU_list.append(best_iou)

            metric.total_elements += len(label_boxes)
            return metric

        global_metrics += compute_metric_boxes(pred_boxes, label_boxes)

    global_metrics.compute_miou()
    global_metrics.compute_accuracy()
    global_metrics.compute_prf()
    print('EVAL --- mIOU : {}, accuracy : {}, precision : {}, '
          'recall : {}, f_measure : {}\n'.format(global_metrics.mIOU, global_metrics.accuracy, global_metrics.precision,
                                                 global_metrics.recall, global_metrics.f_measure))

    return {
        'precision': global_metrics.precision,
        'recall': global_metrics.recall,
        'f_measure': global_metrics.f_measure,
        'mIOU': global_metrics.mIOU
    }