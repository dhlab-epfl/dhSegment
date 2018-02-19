import os
from .base import Metrics
from glob import glob
from scipy.misc import imread, imsave, imresize
import cv2
import numpy as np
from tqdm import tqdm
from .segmentation import compare_bin_prediction_to_label
from ..post_processing.boxes_detection import find_box


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

        pred_box = find_box(np.uint8(bin_upscaled), mode='min_rectangle')
        label_box = find_box(np.uint8(label_image), mode='min_rectangle')

        if debug_folder is not None:
            imsave(os.path.join(debug_folder, '{}_bin.png'.format(basename)), np.uint8(bin_upscaled*255))
            orig_img = imread(os.path.join(validation_dir, 'images', '{}.jpg'.format(basename)))
            cv2.polylines(orig_img, [label_box[:, None, :]], True, (0, 255, 0), thickness=15)
            if pred_box is not None:
                cv2.polylines(orig_img, [pred_box[:, None, :]], True, (0, 0, 255), thickness=15)
            imsave(os.path.join(debug_folder, '{}_boxes.jpg'.format(basename)), orig_img)

        def intersection_over_union(cnt1, cnt2):
            mask1 = np.zeros_like(label_image)
            mask1 = cv2.fillConvexPoly(mask1, cnt1.astype(np.int32), 1).astype(np.int8)
            mask2 = np.zeros_like(label_image)
            mask2 = cv2.fillConvexPoly(mask2, cnt2.astype(np.int32), 1).astype(np.int8)
            return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        if pred_box is not None:
            iou = intersection_over_union(label_box[:, None, :], pred_box[:, None, :])
            global_metrics.IOU_list.append(iou)
        else:
            global_metrics.IOU_list.append(0)
            if verbose:
                print('No box found for {}'.format(basename))

    if pixel_wise:
        # global_metrics.compute_mse()
        # global_metrics.compute_psnr()
        global_metrics.compute_prf()

        print('EVAL --- R : {}, P : {}, FM : {}\n'.format(global_metrics.recall,
                                                        global_metrics.precision, global_metrics.f_measure))

    global_metrics.compute_miou()
    print('EVAL --- mIOU : {}\n'.format(global_metrics.mIOU))
    # Export txt similar to test txt ?

    return {
        'precision': global_metrics.precision,
        'recall': global_metrics.recall,
        'f_measure': global_metrics.f_measure,
        'mIOU': global_metrics.mIOU
    }
