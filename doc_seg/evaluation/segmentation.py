import os
from .base import Metrics
from glob import glob
from scipy.misc import imread, imsave
import cv2
import numpy as np
from tqdm import tqdm


DIVA_CLASSES = {
    -1: 'background',
    0: 'main_text',
    1: 'decorations',
    2: 'comments'
                }


def dibco_evaluate_folder(output_folder: str, validation_dir: str, verbose=False, debug_folder=None) -> dict:
    filenames_processed = glob(os.path.join(output_folder, '*.png'))

    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    global_metrics = Metrics()
    for filename in filenames_processed:
        post_processed = imread(filename, mode='L')
        post_processed = post_processed / np.max(post_processed)

        basename = os.path.basename(filename).split('.')[0]
        label_image = imread(os.path.join(validation_dir, 'labels', '{}.png'.format(basename)), mode='L')
        label_image_normalized = label_image / np.max(label_image)

        target_shape = (label_image.shape[1], label_image.shape[0])
        bin_upscaled = cv2.resize(np.uint8(post_processed), target_shape, interpolation=cv2.INTER_NEAREST)

        # Compute errors
        metric = compare_bin_prediction_to_label(bin_upscaled, label_image_normalized)
        global_metrics += metric

        if debug_folder is not None:
            debug_image = np.zeros((*label_image.shape[:2], 3), np.uint8)
            debug_image[np.logical_and(bin_upscaled, label_image_normalized)] = [0, 255, 0]
            debug_image[np.logical_and(bin_upscaled, label_image_normalized == 0)] = [0, 0, 255]
            debug_image[np.logical_and(bin_upscaled == 0, label_image_normalized)] = [255, 0, 0]
            imsave(os.path.join(debug_folder, basename + '.png'), debug_image)

    global_metrics.compute_mse()
    global_metrics.compute_psnr()
    global_metrics.compute_prf()

    if verbose:
        print('EVAL --- PSNR : {}, R : {}, P : {}, FM : {}'.format(global_metrics.PSNR, global_metrics.recall,
                                                                   global_metrics.precision, global_metrics.f_measure))

    return {k: v for k, v in vars(global_metrics).items() if k in ['MSE', 'PSNR', 'precision', 'recall', 'f_measure']}


def compare_bin_prediction_to_label(prediction: np.array, gt_image: np.array):
    """
    Compares the prediction with the groundtruth.
    :param prediction: prediction (after binarization) should not be probabilities but labels
    :param gt_image: gt image with labels
    :return:
    """
    metrics = Metrics()

    metrics.SE_list = np.sum((gt_image - prediction) ** 2)
    metrics.total_elements = np.size(prediction)

    metrics.true_positives = np.sum(np.logical_and(gt_image, prediction))
    metrics.false_positives = np.sum(np.logical_and(np.logical_xor(gt_image, prediction),
                                                    prediction))
    metrics.false_negatives = np.sum(np.logical_and(np.logical_xor(gt_image, prediction),
                                                    gt_image))
    metrics.true_negatives = np.sum(np.logical_and(np.logical_not(gt_image), np.logical_not(prediction)))

    return metrics


def diva_evaluate_folder(output_folder: str, validation_dir: str, command_line_tool=True,
                         debug_folder: str = None, verbose: bool = False) -> dict:
    # TODO (do not forget to take several classes into account)
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    filenames_post_processed = glob(os.path.join(output_folder, '*.png'))

    n_channels = 3
    global_metrics_per_class = {DIVA_CLASSES[ch]: Metrics() for ch in range(n_channels)}

    for filename in tqdm(filenames_post_processed, desc='Evaluation'):
        basename = os.path.basename(filename).split('.')[0]

        # Open post_processed and label image
        post_processed_img = imread(filename)
        # post_processed_img = post_processed_img / np.maximum(np.max(post_processed_img), 1)

        label_image = imread(os.path.join(validation_dir, 'labels', '{}.png'.format(basename)))
        # label_image = label_image / np.max(label_image)

        # Upsample processed image to compare it to original image
        target_shape = (label_image.shape[1], label_image.shape[0])
        bin_upscaled = cv2.resize(np.uint8(post_processed_img), target_shape, interpolation=cv2.INTER_NEAREST)

        for ch in range(n_channels):
            metric = compare_bin_prediction_to_label(bin_upscaled[:, :, ch], label_image[:, :, ch])
            global_metrics_per_class[DIVA_CLASSES[ch]] += metric
        # TODO : compute accuracy for background class

    global_metric_all_classes = Metrics()
    for key in global_metrics_per_class.keys():
        global_metrics_per_class[key].compute_mse()
        global_metrics_per_class[key].compute_psnr()
        global_metrics_per_class[key].compute_prf()
        global_metrics_per_class[key].compute_accuracy()
        global_metric_all_classes += global_metrics_per_class[key]

    global_metric_all_classes.compute_mse()
    global_metric_all_classes.compute_psnr()
    global_metric_all_classes.compute_prf()
    global_metric_all_classes.compute_accuracy()

    print('EVAL --- Accuracy :  {}, PSNR : {}, R : {}, P : {}, FM : {}\n'.format(
        global_metric_all_classes.accuracy,
        global_metric_all_classes.psnr,
        global_metric_all_classes.recall,
        global_metric_all_classes.precision,
        global_metric_all_classes.f_measure))

    # There is also a command line tool...
    if command_line_tool:
        pass

    # Todo : Not sure it is the best to have nested dictionnaries
    return {
        'precision': {
            **{'global': global_metric_all_classes.precision},
            **{key: global_metrics_per_class[key].precision for key in global_metrics_per_class.keys()}
        },
        'recall': {
            **{'global': global_metric_all_classes.recall},
            **{key: global_metrics_per_class[key].recall for key in global_metrics_per_class.keys()}
        },
        'f_measure': {
            **{'global': global_metric_all_classes.f_measure},
            **{key: global_metrics_per_class[key].f_measure for key in global_metrics_per_class.keys()}
        },
        'psnr': {
            **{'global': global_metric_all_classes.psnr},
            **{key: global_metrics_per_class[key].psnr for key in global_metrics_per_class.keys()}
        },
        'accuracy': {
            **{'global': global_metric_all_classes.accuracy},
            **{key: global_metrics_per_class[key].accuracy for key in global_metrics_per_class.keys()}
        }
    }