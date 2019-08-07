from typing import Tuple, List
import numpy as np
from scipy.ndimage import label
import cv2
import os
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from imageio import imsave, imread
from scipy.misc import imresize
from dh_segment.utils import dump_pickle
from dh_segment.post_processing.binarization import hysteresis_thresholding, cleaning_probs
from dh_segment.post_processing.line_vectorization import find_lines
from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel


def prediction_fn(model_dir: str,
                  input_dir: str,
                  output_dir: str=None,
                  config: tf.ConfigProto=None) -> None:
    """
    Given a model directory this function will load the model and apply it to the files (.jpg, .png) found in input_dir.
    The predictions will be saved in output_dir as .npy files (values ranging [0,255])

    :param model_dir: Directory containing the saved model
    :param input_dir: input directory where the images to predict are
    :param output_dir: output directory to save the predictions (probability images)
    :param config: ConfigProto object to pass to the session in order to define which GPU to use
    :return:
    """
    if not output_dir:
        # For model_dir of style model_name/export/timestamp/ this will create a folder model_name/predictions'
        output_dir = '{}'.format(os.path.sep).join(model_dir.split(os.path.sep)[:-3] + ['predictions'])

    os.makedirs(output_dir, exist_ok=True)
    filenames_to_predict = glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir, '*.png'))

    with tf.Session(config=config):
        m = LoadedModel(model_dir, predict_mode='filename_original_shape')
        for filename in tqdm(filenames_to_predict, desc='Prediction'):
            pred = m.predict(filename)['probs'][0]
            np.save(os.path.join(output_dir, os.path.basename(filename).split('.')[0]), np.uint8(255 * pred))


def cbad_post_processing_fn(probs: np.array,
                            baseline_chanel: int=1,
                            sigma: float=2.5,
                            low_threshold: float=0.8,
                            high_threshold: float=0.9,
                            filter_width: float=0,
                            vertical_maxima: bool=False,
                            output_basename=None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a probability map, returns the contour of lines and the corresponding mask.
    Saves the results in .pkl file if requested.

    :param probs: output of the model (probabilities) in range [0, 255]
    :param baseline_chanel: channel where the baseline class is detected
    :param sigma: sigma value for gaussian filtering
    :param low_threshold: hysteresis low threshold
    :param high_threshold: hysteresis high threshold
    :param filter_width: percentage of the image width to filter out lines that are close to borders (default 0.0)
    :param output_basename: name of file to save the intermediaty result as .pkl file.
    :param vertical_maxima: set to True to use vertical local maxima as candidates for the hysteresis thresholding
    :return: contours, mask
     WARNING : contours IN OPENCV format List[np.ndarray(n_points, 1, (x,y))]
    """

    contours, lines_mask = line_extraction_v1(probs[:, :, baseline_chanel], sigma, low_threshold, high_threshold,
                                              filter_width, vertical_maxima)
    if output_basename is not None:
        dump_pickle(output_basename+'.pkl', (contours, lines_mask.shape))
    return contours, lines_mask


def line_extraction_v1(probs: np.ndarray,
                       low_threshold: float,
                       high_threshold: float,
                       sigma: float=0.0,
                       filter_width: float=0.00,
                       vertical_maxima: bool=False) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a probability map, returns the contour of lines and the corresponding mask

    :param probs: probability map (numpy array)
    :param low_threshold: hysteresis low threshold
    :param high_threshold: hysteresis high threshold
    :param sigma: sigma value for gaussian filtering
    :param filter_width: percentage of the image width to filter out lines that are close to borders (default 0.0)
    :param vertical_maxima: set to True to use vertical local maxima as candidates for the hysteresis thresholding
    :return:
    """
    # Smooth
    probs2 = cleaning_probs(probs, sigma=sigma)

    lines_mask = hysteresis_thresholding(probs2, low_threshold, high_threshold,
                                         candidates_mask=vertical_local_maxima(probs2) if vertical_maxima else None)
    # Remove lines touching border
    # lines_mask = remove_borders(lines_mask)

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


def vertical_local_maxima(probs: np.ndarray) -> np.ndarray:
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[1:-1] = (probs[1:-1] >= probs[:-2]) & (probs[2:] <= probs[1:-1])
    local_maxima = cv2.morphologyEx(local_maxima.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return local_maxima > 0


def remove_borders(mask: np.ndarray, margin: int=5) -> np.ndarray:
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


def extract_lines(npy_filename: str,
                  output_dir: str,
                  original_shape: list,
                  post_process_params: dict,
                  channel_baselines: int=1,
                  mask_dir: str=None,
                  debug: bool=False):
    """
    From the prediction files (probs) (.npy) finds and extracts the lines into PAGE-XML format.

    :param npy_filename: filename of saved predictions (probs) in range (0,255)
    :param output_dir: output direcoty to save the xml files
    :param original_shape: shpae of the original input image (to rescale the extracted lines if necessary)
    :param post_process_params: pramas for lines detection (sigma, thresholds, ...)
    :param channel_baselines: channel where the baseline class is detected
    :param mask_dir: directory containing masks of the page in order to improve the line extraction
    :param debug: if True will output the binary image of the extracted lines
    :return: contours of lines (open cv format), binary image of lines (lines mask)
    """

    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(npy_filename).split('.')[0]

    pred = np.load(npy_filename)/255  # type: np.ndarray
    lines_prob = pred[:, :, channel_baselines]

    if mask_dir is not None:
        mask = imread(os.path.join(mask_dir, basename + '.png'), mode='L')
        mask = imresize(mask, lines_prob.shape)
        lines_prob[mask == 0] = 0.

    contours, lines_mask = line_extraction_v1(lines_prob, **post_process_params)

    if debug:
        imsave(os.path.join(output_dir, '{}_bin.jpg'.format(basename)), lines_mask)

    ratio = (original_shape[0] / pred.shape[0], original_shape[1] / pred.shape[1])
    xml_filename = os.path.join(output_dir, basename + '.xml')
    PAGE.save_baselines(xml_filename, contours, ratio, predictions_shape=pred.shape[:2])

    return contours, lines_mask
