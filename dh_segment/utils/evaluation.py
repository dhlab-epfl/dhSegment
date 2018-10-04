#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import numpy as np
import json
import cv2


class Metrics:
    def __init__(self):
        self.total_elements = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.SE_list = list()
        self.IOU_list = list()

        self.MSE = 0
        self.psnr = 0
        self.mIOU = 0
        self.IU = 0
        self.accuracy = 0
        self.recall = 0
        self.precision = 0
        self.f_measure = 0

    def __add__(self, other):
        if isinstance(other, self.__class__):
            summable_attr = ['total_elements', 'false_negatives', 'false_positives', 'true_positives', 'true_negatives']
            addlist_attr = ['SE_list', 'IOU_list']
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
        self.MSE = np.sum(self.SE_list) / self.total_elements if self.total_elements > 0 else np.inf
        return self.MSE

    def compute_psnr(self):
        if self.MSE != 0:
            self.psnr = 10 * np.log10((1 ** 2) / self.MSE)
            return self.psnr
        else:
            print('Cannot compute PSNR, MSE is 0.')

    def compute_prf(self, beta=1):
        self.recall = self.true_positives / (self.true_positives + self.false_negatives) \
            if (self.true_positives + self.false_negatives) > 0 else 0
        self.precision = self.true_positives / (self.true_positives + self.false_positives) \
            if (self.true_positives + self.false_negatives) > 0 else 0
        self.f_measure = ((1 + beta ** 2) * self.recall * self.precision) / (self.recall + (beta ** 2) * self.precision) \
            if (self.recall + self.precision) > 0 else 0

        return self.recall, self.precision, self.f_measure

    def compute_miou(self):
        self.mIOU = np.mean(self.IOU_list)
        return self.mIOU

    # See http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2017/DocUsingDeepFeatures.pdf
    def compute_iu(self):
        self.IU = self.true_positives / (self.true_positives + self.false_positives + self.false_negatives) \
            if (self.true_positives + self.false_positives + self.false_negatives) > 0 else 0
        return self.IU

    def compute_accuracy(self):
        self.accuracy = (self.true_positives + self.true_negatives)/self.total_elements if self.total_elements > 0 else 0

    def save_to_json(self, json_filename: str) -> None:
        export_dic = self.__dict__.copy()
        del export_dic['MSE_list']

        with open(json_filename, 'w') as outfile:
            json.dump(export_dic, outfile)


def intersection_over_union(cnt1, cnt2, shape_mask):
    mask1 = np.zeros(shape_mask, np.uint8)
    mask1 = cv2.fillConvexPoly(mask1, cnt1.astype(np.int32), 1).astype(np.int8)
    mask2 = np.zeros(shape_mask, np.uint8)
    mask2 = cv2.fillConvexPoly(mask2, cnt2.astype(np.int32), 1).astype(np.int8)
    return np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
