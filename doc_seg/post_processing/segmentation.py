import numpy as np

import cv2


def dibco_binarization_fn(probs: np.ndarray, threshold=0.5):
    probs = probs[:, :, 1]
    if threshold < 0:
        probs = np.uint8(probs * 255)
        # Otsu's thresholding
        blur = cv2.GaussianBlur(probs, (5, 5), 0)
        thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return np.uint8(bin_img / 255)
    else:
        return (probs > threshold).astype(np.uint8)


def page_post_processing_fn(probs: np.ndarray, threshold=0.5):

    probs = probs > threshold

    probs = cv2.morphologyEx((probs.astype(np.uint8) * 255), cv2.MORPH_OPEN, kernel=np.ones((7, 7)))
    probs = cv2.morphologyEx(probs, cv2.MORPH_CLOSE, kernel=np.ones((9, 9)))

    return probs / 255


def diva_post_processing_fn(predictions):
    # TODO
    return None