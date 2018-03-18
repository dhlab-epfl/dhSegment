import json
import tempfile
import os
import numpy as np
from glob import glob
from tqdm import tqdm


def evaluate_epoch(exported_eval_files_dir, validation_dir: str, post_process_fn, evaluation_fn,
                   post_process_params=dict(), verbose: bool=False, debug_folder=None):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Process predicted probs
        filenames_exported_predictions = glob(os.path.join(exported_eval_files_dir, '*.npy'))
        for filename in tqdm(filenames_exported_predictions, desc='Post-processing'):
            basename = os.path.basename(filename).split('.')[0]

            predictions = np.load(filename)
            predictions_normalized = predictions / 255

            post_process_fn(predictions_normalized, output_basename=os.path.join(tmpdirname, basename),
                            **post_process_params)

        return evaluation_fn(tmpdirname, validation_dir, debug_folder=debug_folder)


class Metrics:
    def __init__(self):
        # TODO : either compute false/true positives/negatives
        # TODO : or keep track of flatten stacked prediction with flatten stacked gt
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


def format_quad_to_string(quad):
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


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
