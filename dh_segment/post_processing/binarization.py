import numpy as np
import cv2
from scipy.ndimage import label


def thresholding(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array in range [0, 1] of shape HxWx2
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """
    if threshold < 0:  # Otsu's thresholding
        probs = np.uint8(probs * 255)
        #TODO Correct that weird gaussianBlur
        probs = cv2.GaussianBlur(probs, (5, 5), 0)

        thresh_val, bin_img = cv2.threshold(probs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = np.uint8(bin_img / 255)
    else:
        mask = np.uint8(probs > threshold)

    return mask


def cleaning_binary(mask: np.ndarray, size: int=5):

    ksize_open = (size, size)
    ksize_close = (size, size)
    mask = cv2.morphologyEx((mask.astype(np.uint8, copy=False) * 255), cv2.MORPH_OPEN, kernel=np.ones(ksize_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones(ksize_close))
    return mask / 255


def hysteresis_thresholding(probs: np.array, low_threshold: float, high_threshold: float,
                            candidates_mask: np.ndarray=None):
    low_mask = probs > low_threshold
    if candidates_mask is not None:
        low_mask = candidates_mask & low_mask
    # Connected components extraction
    label_components, count = label(low_mask, np.ones((3, 3)))
    # Keep components with high threshold elements
    good_labels = np.unique(label_components[low_mask & (probs > high_threshold)])
    label_masks = np.zeros((count + 1,), bool)
    label_masks[good_labels] = 1
    return label_masks[label_components]


def cleaning_probs(probs: np.ndarray, sigma: float):
    # Smooth
    if sigma > 0.:
        return cv2.GaussianBlur(probs, (int(3*sigma)*2+1, int(3*sigma)*2+1), sigma)
    elif sigma == 0.:
        return cv2.fastNlMeansDenoising((probs*255).astype(np.uint8), h=20)/255
    else:  # Negative sigma, do not do anything
        return probs
