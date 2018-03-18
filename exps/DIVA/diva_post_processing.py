#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
import cv2
from scipy.misc import imsave, imread
from typing import List
import os


def diva_post_processing_fn(probs: np.array, thresholds: List[float]=[0.5, 0.5, 0.5], min_cc: int=0,
                            page_mask: np.array=None, output_basename: str=None) -> np.ndarray:
    """

    :param probs: array in range [0, 1] of shape HxWx3
    :param thresholds: list of length 3 corresponding to the threshold for each channel
    :param min_cc: minimum size of connected components to keep
    :param border_removal: removes pixels in left and right border of the image that are within a certain margin
    :param output_basename:
    :return:
    """
    # border_margin = probs.shape[1] * 0.02
    final_mask = np.zeros_like(probs, dtype=np.uint8)
    # Compute binary mask for each class (each channel)

    if page_mask is not None:
        probs = (page_mask > 0)[:, :, None] * probs

    for ch in range(probs.shape[-1]):
        probs_ch = probs[:, :, ch]
        if thresholds[ch] < 0:  # Otsu thresholding
            probs_ch = np.uint8(probs_ch * 255)
            blur = cv2.GaussianBlur(probs_ch, (5, 5), 0)
            thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin_img = bin_img / 255
        else:
            bin_img = probs_ch > thresholds[ch]

        if min_cc > 0:
            _, labeled_cc = cv2.connectedComponents(bin_img.astype(np.uint8), connectivity=8)
            for lab in np.unique(labeled_cc):
                mask = labeled_cc == lab
                if np.sum(mask) < min_cc:
                    labeled_cc[mask] = 0
            final_mask[:, :, ch] = bin_img * (labeled_cc > 0)
        else:
            final_mask[:, :, ch] = bin_img

    result = final_mask.astype(int)

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)

    return result
