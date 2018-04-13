from typing import Tuple, List
import numpy as np
from scipy.ndimage import label

import cv2
import os
from dh_segment.utils import dump_pickle
from dh_segment.post_processing.binarization import hysteresis_thresholding, cleaning_probs
from dh_segment.post_processing.line_vectorization import find_lines


def cbad_post_processing_fn(probs: np.array, sigma: float=2.5, low_threshold: float=0.8, high_threshold: float=0.9,
                            filter_width: float=0, output_basename=None):
    """

    :param probs: output of the model (probabilities) in range [0, 255]
    :param filename: filename of the image processed
    :param xml_output_dir: directory to export the resulting PAGE XML
    :param upsampled_shape: shape of the original image
    :param sigma:
    :param low_threshold:
    :param high_threshold:
    :return: contours, mask
     WARNING : contours IN OPENCV format List[np.ndarray(n_points, 1, (x,y))]
    """

    contours, lines_mask = line_extraction_v1(probs[:, :, 1], sigma, low_threshold, high_threshold, filter_width)
    if output_basename is not None:
        dump_pickle(output_basename+'.pkl', (contours, lines_mask.shape))
    return contours, lines_mask


def line_extraction_v0(probs, sigma, threshold):
    # probs_line = probs[:, :, 1]
    probs_line = probs
    # Smooth
    probs2 = cv2.GaussianBlur(probs_line, (int(3*sigma)*2+1, int(3*sigma)*2+1), sigma)

    lines_mask = probs2 >= threshold
    # Extract polygons from line mask
    contours = find_lines(lines_mask)

    return contours, lines_mask


def line_extraction_v1(probs, low_threshold, high_threshold, sigma=0.0, filter_width=0.00, vertical_maxima=False):
    probs_line = probs
    # Smooth
    probs2 = cleaning_probs(probs, sigma=sigma)

    lines_mask = hysteresis_thresholding(probs2, low_threshold, high_threshold,
                                         candidates=vertical_local_maxima(probs2) if vertical_maxima else None)
    # Remove lines touching border
    #lines_mask = remove_borders(lines_mask)
    # Extract polygons from line mask
    contours = find_lines(lines_mask)

    filtered_contours = []
    page_width = probs.shape[1]
    for cnt in contours:
        centroid_x, centroid_y = np.mean(cnt, axis=0)[0]
        if centroid_x < filter_width*page_width or centroid_x > (1-filter_width)*page_width:
            continue
        # if cv2.arcLength(cnt, False) < filter_width*page_width:
        #    continue
        filtered_contours.append(cnt)

    return filtered_contours, lines_mask


# TODO Is this still updated ?
# def line_extraction_v2(probs, sigma, low_threshold, high_threshold):
#     probs_line = probs
#     # Smooth
#     probs2 = cv2.GaussianBlur(probs_line, (int(3*sigma)*2+1, int(3*sigma)*2+1), sigma)
#     seeds = probs2 > high_threshold
#     labelled_components, nb_components = label(seeds)
#
#     lines_mask = hysteresis_thresholding(probs2, local_maxima, low_threshold, high_threshold)
#     # Remove lines touching border
#     #lines_mask = remove_borders(lines_mask)
#     # Extract polygons from line mask
#     contours = extract_line_polygons(lines_mask)
#
#     filtered_contours = []
#     page_width = probs.shape[1]
#     for cnt in contours:
#         if cv2.arcLength(cnt, False) < 0.05*page_width:
#             continue
#         if cv2.arcLength(cnt, False) < 0.05*page_width:
#             continue
#         filtered_contours.append(cnt)
#
#     return filtered_contours, lines_mask


def vertical_local_maxima(probs):
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[1:-1] = (probs[1:-1] >= probs[:-2]) & (probs[2:] <= probs[1:-1])
    local_maxima = cv2.morphologyEx(local_maxima.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return local_maxima > 0


def upscale_coordinates(list_points: List[np.array], ratio: Tuple[float, float]):  # list of (N,1,2) cv2 points
    return np.array(
        [(round(p[0, 0]*ratio[1]), round(p[0, 1]*ratio[0])) for p in list_points]
    )[:, None, :].astype(int)


def get_image_basename(image_filename: str, with_acronym: bool=False):
    # Get acronym followed by name of file
    directory, basename = os.path.split(image_filename)
    if with_acronym:
        acronym = directory.split(os.path.sep)[-1].split('_')[0]
        return '{}_{}'.format(acronym, basename.split('.')[0])
    else:
        return '{}'.format(basename.split('.')[0])


def get_page_filename(image_filename):
    return os.path.join(os.path.dirname(image_filename), 'page', '{}.xml'.format(os.path.basename(image_filename)[:-4]))


def remove_borders(mask, margin=5):
    tmp = mask.copy()
    tmp[:margin] = 1
    tmp[-margin:] = 1
    tmp[:, :margin] = 1
    tmp[:, -margin:] = 1
    label_components, count = label(tmp, np.ones((3, 3)))
    result = mask.copy()
    border_component = label_components[0, 0]
    result[label_components == border_component] = 0
    return result