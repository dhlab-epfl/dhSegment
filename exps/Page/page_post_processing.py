#!/usr/bin/env python
__author__ = 'solivr'

import numpy as np
from scipy.misc import imsave
from dh_segment.post_processing import binarization


def page_post_processing_fn(probs: np.ndarray, threshold: float=0.5, output_basename: str=None,
                            ksize_open: tuple=(7, 7), ksize_close: tuple=(9, 9), kernel_size:int = 5) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :param output_basename:
    :param ksize_open: size of kernel for morphological opening
    :param ksize_close: size of kernel for morphological closing
    :return: binary mask
    """

    mask = binarization.thresholding(probs[:, :, 1], threshold=threshold)
    result = binarization.cleaning_binary(mask, size=kernel_size)

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)
    return result