#!/usr/bin/env python
__author__ = 'solivr'

import cv2
import numpy as np
from scipy.misc import imsave


def ornaments_post_processing_fn(probs: np.ndarray, threshold: float=0.5, ksize_open: tuple=(5, 5),
                                 ksize_close: tuple=(7, 7), output_basename: str=None) -> np.ndarray:
    """

    :param probs:
    :param threshold:
    :param ksize_open:
    :param ksize_close:
    :param output_basename:
    :return:
    """
    probs = probs[:, :, 1]
    if threshold < 0:  # Otsu thresholding
        probs_ch = np.uint8(probs * 255)
        blur = cv2.GaussianBlur(probs_ch, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = bin_img / 255
    else:
        mask = probs > threshold
        # TODO : adaptive kernel (not hard-coded)
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones(ksize_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones(ksize_close))

    result = mask / 255

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)
    return result
