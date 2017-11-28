import numpy as np
from scipy.ndimage import label
from typing import Tuple, List


def vertical_local_maxima(probs):
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[1:-1] = (probs[1:-1] > probs[:-2]) & (probs[2:] < probs[1:-1])
    return local_maxima


def hysteresis_thresholding(probs: np.array, candidates: np.array, low_threshold: float, high_threshold: float):
    low_mask = candidates & (probs > low_threshold)
    # Connected components extraction
    label_components, count = label(low_mask, np.ones((3, 3)))
    # Keep components with high threshold elements
    good_labels = np.unique(label_components[low_mask & (probs > high_threshold)])
    label_masks = np.zeros((count + 1,), bool)
    label_masks[good_labels] = 1
    return label_masks[label_components]


def upscale_coordinates(list_points: List[np.array], ratio: Tuple[float, float]):  # list of (N,1,2) cv2 points
    return np.array(
        [(round(p[0, 0]*ratio[1]), round(p[0, 1]*ratio[0])) for p in list_points]
    )[:, None, :].astype(int)
