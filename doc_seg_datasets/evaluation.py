#!/usr/bin/env python
__author__ = 'solivr'

from glob import glob
from scipy.misc import imread
import numpy as np
import os
import cv2
import json
import sklearn.metrics as skmetrics
from .post_processing import cbad_post_processing_fn, dibco_binarization_fn, page_post_processing_fn


CBAD_JAR = '/scratch/dataset/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
# INTERPOLATION_DICT = {'BILINEAR': cv2.INTER_LINEAR,
#                       'NEAREST': cv2.INTER_NEAREST,
#                       'CUBIC': cv2.INTER_CUBIC}

class Metrics:
    def __init__(self):
        # TODO : either compute false/true positives/negatives
        # TODO : or keep track of flatten stacked prediction with flatten stacked gt
        self.total_elements = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.SE_list = list()

        self.MSE = 0
        self.PSNR = 0
        self.recall = 0
        self.precision = 0
        self.f_measure = 0

    def __add__(self, other):
        if isinstance(other, self.__class__):
            summable_attr = ['total_elements', 'false_negatives', 'false_positives', 'true_positives']
            addlist_attr = ['SE_list']
            m = Metrics()
            for k, v in self.__dict__.items():
                if k in summable_attr:
                    setattr(m, k, self.__dict__[k] + other.__dict__[k])
                elif k in addlist_attr:
                    mse1 = [self.__dict__[k]] if not isinstance(self.__dict__[k], list) else self.__dict__[k]
                    mse2 = [other.__dict__[k]] if not isinstance(other.__dict__[k], list) else other.__dict__[k]

                    setattr(m, k, mse1 + mse2)
            return m
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self.__add__(other)

    def compute_mse(self):
        self.MSE = np.sum(self.SE_list) / self.total_elements
        return self.MSE

    def compute_psnr(self):
        if self.MSE != 0:
            self.PSNR = 10 * np.log10((1 ** 2) / self.MSE)
            return self.PSNR
        else:
            print('Cannot compute PSNR, MSE is 0.')

    def compute_prf(self, beta=1):
        # Todo use scikit learn
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.f_measure = ((1 + beta**2) * self.recall * self.precision) / (self.recall + (beta**2)*self.precision)

        return self.recall, self.precision, self.f_measure


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

    return metrics


def save_metrics_to_json(metric: Metrics, json_filename: str) -> None:
    export_dic = metric.__dict__
    del export_dic['MSE_list']

    with open(json_filename, 'w') as outfile:
        json.dump(export_dic, outfile)


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


def dibco_evaluate_epoch(exported_eval_files_dir: str, validation_labels_dir: str, verbose:bool=False, **kwargs):
    """

    :param exported_eval_files_dir:
    :param validation_labels_dir:
    :param kwargs:
    :return:
    """

    filenames_exported_predictions = glob(os.path.join(exported_eval_files_dir, '*.npy'))

    global_metrics = Metrics()
    for filename in filenames_exported_predictions:
        basename = os.path.basename(filename).split('.')[0]

        # Open label image
        label_image = imread(os.path.join(validation_labels_dir, '{}.png'.format(basename)), mode='L')
        label_image_normalized = label_image / np.max(label_image)

        predictions = np.load(filename)[:, :, 1]
        predictions_normalized = predictions / 255

        # Post processing (results should be in range [0, 255]!)
        processed_bin = dibco_binarization_fn(predictions_normalized, **kwargs)
        target_shape = (label_image.shape[1], label_image.shape[0])
        bin_normalized = cv2.resize(np.uint8(processed_bin), target_shape, interpolation=cv2.INTER_NEAREST)

        # Compute errors
        metric = compare_bin_prediction_to_label(bin_normalized, label_image_normalized)
        global_metrics += metric

    global_metrics.compute_mse()
    global_metrics.compute_psnr()
    global_metrics.compute_prf()

    if verbose:
        print('EVAL --- PSNR : {}, R : {}, P : {}, FM : {}'.format(global_metrics.PSNR, global_metrics.recall,
                                                                   global_metrics.precision, global_metrics.f_measure))

    # save_metrics_to_json(global_metrics, os.path.join(exported_files_dir, 'validation_scores.json'))
    return global_metrics


def evaluate_cbad(exported_files_dir: str, validation_labels_dir: str, jar_path: str=CBAD_JAR, **kwargs) -> None:
    """

    :param exported_files_dir:
    :param validation_labels_dir:
    :param jar_path:
    :param kwargs: 'xml_output_dir', 'sigma', 'low_threshold', 'high_threshold'
    :return:
    """

    filenames_exported_predictions = glob(os.path.join(exported_files_dir, '*.npy'))

    xml_filenames_list = list()
    for filename in filenames_exported_predictions:
        # Open label image
        label_image = imread(os.path.join(validation_labels_dir,
                                          '{}.png'.format(os.path.basename(filename).split('.')[0])), mode='L')

        predictions = np.load(filename)[:, :, 1]
        predictions_normalized = predictions / 255

        xml_filenames = cbad_post_processing_fn(predictions_normalized, filename,
                                                upsampled_shape=label_image.shape[:2] **kwargs)
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
    # There is also a command line tool...


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
