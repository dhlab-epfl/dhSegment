#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
from scipy.misc import imsave
import cv2


def dibco_binarization_fn(probs: np.ndarray, threshold=0.5, output_basename=None):
    probs = probs[:, :, 1]
    if threshold < 0:
        probs = np.uint8(probs * 255)
        # Otsu's thresholding
        blur = cv2.GaussianBlur(probs, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = np.uint8(bin_img / 255)
    else:
        result = (probs > threshold).astype(np.uint8)

    if output_basename is not None:
        imsave(output_basename + '.png', result*255)
    return result