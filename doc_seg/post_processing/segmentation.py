import numpy as np

import cv2
from scipy.misc import imsave


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


def page_post_processing_fn(probs: np.ndarray, threshold: float=0.5, output_basename: str=None,
                            ksize_open: tuple=(7, 7), ksize_close: tuple=(9, 9)) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWxC
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :param output_basename:
    :param ksize_open: size of kernel for morphological opening
    :param ksize_close: size of kernel for morphological closing
    :return: binary mask
    """
    probs = probs[:, :, 1]
    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        blur = cv2.GaussianBlur(probs, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.uint8(bin_img / 255)
    else:
        mask = probs > threshold
    # TODO : adaptive kernel (not hard-coded)
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones(ksize_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones(ksize_close))

    result = mask / 255

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)
    return result


def diva_post_processing_fn(predictions):
    # TODO
    return None