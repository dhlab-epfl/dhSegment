import numpy as np
from scipy.misc import imsave
from typing import List
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


def page_post_processing_fn(probs: np.ndarray, threshold: float=0.5, output_basename: str=None,
                            ksize_open: tuple=(7, 7), ksize_close: tuple=(9, 9)) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
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


def diva_post_processing_fn(probs: np.ndarray, thresholds: List[float]=[0.5, 0.5, 0.5], min_cc: int=0,
                            border_removal: bool=False, output_basename: str=None) -> np.ndarray:
    """

    :param probs: array in range [0, 1] of shape HxWx3
    :param thresholds: list of length 3 corresponding to the threshold for each channel
    :param min_cc: minimum size of connected components to keep
    :param border_removal: removes pixels in left and right border of the image that are within a certain margin
    :param output_basename:
    :return:
    """
    border_margin = probs.shape[1] * 0.02
    final_mask = np.zeros_like(probs, dtype=np.uint8)
    # Compute binary mask for each class (each channel)
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

        if border_removal:
            final_mask[:, :border_margin, ch] = 0
            final_mask[:, -border_margin:, ch] = 0

    result = final_mask.astype(int)

    if output_basename is not None:
        imsave('{}.png'.format(output_basename), result*255)

    return result
