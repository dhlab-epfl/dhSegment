import os
from .base import Metrics
from glob import glob
from scipy.misc import imread, imsave, imresize
import cv2
import numpy as np


def cini_evaluate_folder(output_folder: str, validation_dir: str, verbose=False, debug_folder=None) -> dict:
    filenames_processed = glob(os.path.join(output_folder, '*.png'))

    iou_cardboards = []
    iou_images = []
    for filename in filenames_processed:
        post_processed = imread(filename)
        post_processed = post_processed / np.max(post_processed)

        # Perform post-process
        predictions = np.load(filename)
        predictions_normalized = predictions / 255  # type: np.ndarray
        cardboard_coords, image_coords = cini_post_processing_fn(predictions_normalized, **kwargs)

        # Open label image
        label_path = os.path.join(validation_dir, 'labels', '{}.png'.format(basename))
        if not os.path.exists(label_path):
            label_path = label_path.replace('.png', '.jpg')
        label_image = imread(label_path, mode='RGB')
        label_image = imresize(label_image, predictions.shape[:2])
        label_predictions = np.stack([
            label_image[:, :, 0] > 250,
            label_image[:, :, 1] > 250,
            label_image[:, :, 2] > 250
        ], axis=-1).astype(np.float32)
        label_cardboard_coords, label_image_coords = cini_post_processing_fn(label_predictions, **kwargs)

        # Compute errors
        def intersection_over_union(cnt1, cnt2):
            mask1 = np.zeros_like(predictions[:, :, 0])
            cv2.fillConvexPoly(mask1, cv2.boxPoints(cnt1).astype(np.int32), 1)
            mask2 = np.zeros_like(predictions[:, :, 0])
            cv2.fillConvexPoly(mask2, cv2.boxPoints(cnt2).astype(np.int32), 1)
            return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        iou_cardboard = intersection_over_union(label_cardboard_coords, cardboard_coords)
        iou_image = intersection_over_union(label_image_coords, image_coords)

        iou_cardboards.append(iou_cardboard)
        iou_images.append(iou_image)

    result = {
        'cardboard_mean_iou': np.mean(iou_cardboards),
        'image_mean_iou': np.mean(iou_images),
    }

    if verbose:
        print(result)


def find_box_prediction(predictions, min_rect=True):
    contours, hierarchy = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_boxes = list()
    if min_rect:
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # Todo : test if it seems a valid box

            list_boxes.append(np.int0(box))
    else:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            # Todo : test if it seems a valid box

            list_boxes.append(box)


# def evaluate_page(exported_files_dir: str, validation_labels_dir: str, pixel_wise=True):
#     # TODO
#
#     filenames_exported_predictions = glob(os.path.join(exported_files_dir, '*.npy'))
#
#     global_metrics = Metrics()
#     for filename in filenames_exported_predictions:
#         basename = os.path.basename(filename).split('.')[0]
#
#         # Open label image
#         label_image = imread(os.path.join(validation_labels_dir, '{}.png'.format(basename)), mode='L')
#         label_image_normalized = label_image / np.max(label_image)
#
#         predictions = np.load(filename)[:, :, 1]
#         predictions_normalized = norm_and_upsample(predictions, target_shape=label_image.shape[:2])
#
#         processed_preds = page_post_processing_fn(predictions_normalized)
#
#         if pixel_wise:
#             metric = compare_bin_prediction_to_label(processed_preds, label_image_normalized)
#             global_metrics += metric
#         else:
#             # TODO rectangles IoUs
#             # Todo : deal with possibility of false positives
#             contours, hierarchy = cv2.findContours(processed_preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for c in contours:
#                 rect = cv2.minAreaRect(c)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)
#             pass
#
#     if pixel_wise:
#         global_metrics.compute_mse()
#         global_metrics.compute_psnr()
#         global_metrics.compute_prf()
#
#         print('EVAL --- PSNR : {}, R : {}, P : {}, FM : {}'.format(global_metrics.PSNR, global_metrics.recall,
#                                                                    global_metrics.precision, global_metrics.f_measure))
#         save_metrics_to_json(global_metrics, os.path.join(exported_files_dir, 'result_metrics.json'))
#     else:
#         # Todo
#         pass
#         # Export txt similar to test txt
#
