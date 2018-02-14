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


def page_post_processing_fn(probs: np.ndarray, threshold=0.5, output_basename=None):

    mask = probs > threshold
    # TODO : adaptive kernel (not hard-coded)
    mask = cv2.morphologyEx((mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones((7, 7)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((9, 9)))

    result = mask / 255

    if output_basename is not None:
        imsave(output_basename + '.png', result*255)
    return result


def diva_post_processing_fn(predictions):
    # TODO
    return None