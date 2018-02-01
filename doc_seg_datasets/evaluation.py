#!/usr/bin/env python
__author__ = 'solivr'

from glob import glob
from scipy.misc import imread
import numpy as np
import os
import cv2
from .post_processing import cbad_post_processing_fn, dibco_binarization_fn, page_post_processing_fn


CBAD_JAR = '/scratch/dataset/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
INTER_DICT = {'BILINEAR': cv2.INTER_LINEAR,
              'NEAREST': cv2.INTER_NEAREST,
              'CUBIC': cv2.INTER_CUBIC}


class Metrics:
    def __init__(self):
        self.total_elements = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.MSE_list = list()

        self.MSE = 0
        self.PSNR = 0
        self.recall = 0
        self.precision = 0
        self.f_measure = 0

    def __add__(self, other):
        if isinstance(other, self.__class__):
            sommable_attr = ['total_elements', 'false_negatives', 'false_positives', 'true_positives']
            addlist_attr = ['MSE_list']
            m = Metrics()
            for k, v in self.__dict__.items():
                if k in sommable_attr:
                    setattr(m, k, self.__dict__[k] + other.__dict__[k])
                elif k in addlist_attr:
                    mse1 = [self.__dict__[k]] if not isinstance(self.__dict__[k], list) else self.__dict__[k]
                    mse2 = [other.__dict__[k]] if not isinstance(other.__dict__[k], list) else other.__dict__[k]

                    print(mse1, mse2)
                    setattr(m, k, mse1 + mse2)
            return m
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def compute_mse(self):
        self.MSE = np.sum(self.MSE_list) / self.total_elements
        return self.MSE

    def compute_psnr(self):
        if self.MSE != 0:
            self.PSNR = 10 * np.log10((1 ** 2) / self.MSE)
            return self.PSNR
        else:
            print('Cannot compute PSNR, MSE is 0.')

    def compute_prf(self, beta=1):
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.f_measure = ((1 + beta**2) * self.recall * self.precision) / (self.recall + (beta**2)*self.precision)

        return self.recall, self.precision, self.f_measure


def compare_prediction_to_label(prediction: np.array, gt_image: np.array):
    """
    Compares the prediction with the groundtruth.
    :param prediction: prediction (after binarization) should not be probabilities but labels
    :param gt_image: gt image with labels
    :return:
    """
    metrics = Metrics()

    metrics.MSE_list = np.sum((gt_image - prediction) ** 2)
    metrics.total_elements = np.size(prediction)

    metrics.true_positives = np.sum(np.logical_and(gt_image, prediction))
    metrics.false_positives = np.sum(np.logical_and(np.logical_xor(gt_image, prediction),
                                                    prediction))
    metrics.false_negatives = np.sum(np.logical_and(np.logical_xor(gt_image, prediction),
                                                    gt_image))

    return metrics


def norm_and_upsample_predictions_image(predictions: np.array, target_shape: tuple, interpolation='BILINEAR') -> np.array:
    """

    :param predictions:
    :param target_shape:
    :param interpolation:
    :return:
    """

    predictions_normalized = predictions / 255
    predictions_normalized = cv2.resize(predictions_normalized, target_shape, interpolation=INTER_DICT[interpolation])

    return predictions_normalized


def save_metrics_to_json(metric: Metrics):
    # TODO
    pass


def evaluate_dibco(exported_files_dir: str, validation_labels_dir: str, **kwargs):
    """

    :param exported_files_dir:
    :param validation_labels_dir:
    :param kwargs:
    :return:
    """

    filenames_exported_predictions = glob(os.path.join(exported_files_dir, '*.npy'))

    global_metrics = Metrics()
    for filename in filenames_exported_predictions:
        basename = os.path.basename(filename).split('.')[0]

        # Open label image
        label_image = imread(os.path.join(validation_labels_dir, '{}.png'.format(basename)), mode='L')
        label_image_normalized = label_image / np.max(label_image)

        predictions = np.load(filename)[:, :, 1]
        predictions_normalized = norm_and_upsample_predictions_image(predictions, target_shape=label_image.shape[:2])

        # Post processing (results should be normalized [0,1]!)
        processed_preds = dibco_binarization_fn(predictions_normalized, **kwargs)

        # Compute errors
        metric = compare_prediction_to_label(processed_preds, label_image_normalized)
        global_metrics += metric

    global_metrics.compute_mse()
    global_metrics.compute_psnr()
    global_metrics.compute_prf()

    print('EVAL --- PSNR : {}, R : {}, P : {}, FM : {}'.format(global_metrics.PSNR, global_metrics.recall,
                                                               global_metrics.precision, global_metrics.f_measure))

    save_metrics_to_json(global_metrics)
    return global_metrics


def evaluate_cbad(exported_files_dir: str, validation_labels_dir: str, jar_path: str=CBAD_JAR, **kwargs) -> None:
    """

    :param exported_files_dir:
    :param validation_labels_dir:
    :param jar_path:
    :param kwargs:
    :return:
    """

    filenames_exported_predictions = glob(os.path.join(exported_files_dir, '*.npy'))

    xml_filenames_list = list()
    for filename in filenames_exported_predictions:
        # Open label image
        label_image = imread(os.path.join(validation_labels_dir,
                                          '{}.png'.format(os.path.basename(filename).split('.')[0])), mode='L')

        predictions = np.load(filename)[:, :, 1]
        predictions_normalized = norm_and_upsample_predictions_image(predictions, target_shape=label_image.shape[:2])

        xml_filenames = cbad_post_processing_fn(predictions_normalized, **kwargs)
        xml_filenames_list.append(xml_filenames)

    gt_pages_list_filename = os.path.join(exported_files_dir, 'gt_pages.lst')
    genereated_pages_list_filename = os.path.join(exported_files_dir, 'generated_pages.lst')
    with open(gt_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[0] for s in xml_filenames_list]))
    with open(genereated_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[1] for s in xml_filenames_list]))

    # Run command line evaluation tool
    os.system('java -jar {} {} {}'.format(jar_path, gt_pages_list_filename, genereated_pages_list_filename))


def evaluate_diva():
    # TODO (do not forget to take several classes into account)
    pass


def evaluate_page(exported_files_dir: str, validation_labels_dir: str, pixel_wise=True):
    # TODO

    filenames_exported_predictions = glob(os.path.join(exported_files_dir, '*.npy'))

    global_metrics = Metrics()
    for filename in filenames_exported_predictions:
        basename = os.path.basename(filename).split('.')[0]

        # Open label image
        label_image = imread(os.path.join(validation_labels_dir, '{}.png'.format(basename)), mode='L')
        label_image_normalized = label_image / np.max(label_image)

        predictions = np.load(filename)[:, :, 1]
        predictions_normalized = norm_and_upsample_predictions_image(predictions, target_shape=label_image.shape[:2])

        processed_preds = page_post_processing_fn(predictions_normalized, pixel_wise)

        if pixel_wise:
            metric = compare_prediction_to_label(processed_preds, label_image_normalized)
            global_metrics += metric
        else:
            # TODO rectangles IoUs
            pass

    if pixel_wise:
        global_metrics.compute_mse()
        global_metrics.compute_psnr()
        global_metrics.compute_prf()

        print('EVAL --- PSNR : {}, R : {}, P : {}, FM : {}'.format(global_metrics.PSNR, global_metrics.recall,
                                                                   global_metrics.precision, global_metrics.f_measure))
        save_metrics_to_json(global_metrics)
    else:
        # Todo
        pass
        # Export txt similar to test txt

