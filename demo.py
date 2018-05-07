#!/usr/bin/env python

import tensorflow as tf
from dh_segment.loader import LoadedModel
from dh_segment.post_processing import boxes_detection, binarization, PAGE
from tqdm import tqdm
from glob import glob
import numpy as np
import os
import cv2
from scipy.misc import imread, imsave

# To output results in PAGE XML format (http://www.primaresearch.org/schema/PAGE/gts/pagecontent/2013-07-15/)
PAGE_XML_DIR = './page_xml'


def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.thresholding(probs, threshold)
    mask = binarization.cleaning_binary(mask, size=5)
    return mask


def format_quad_to_string(quad):
    """
    Formats the corner points into a string.
    :param quad: coordinates of the quadrilateral
    :return:
    """
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


if __name__ == '__main__':

    # If the model has been trained load the model, otherwise use the given model
    model_dir = 'demo/page_model/export'
    if not os.path.exists(model_dir):
        model_dir = 'demo/model/'

    input_files = glob('demo/pages/test_a1/images/*')

    output_dir = 'demo/processed_images'
    os.makedirs(output_dir, exist_ok=True)
    # PAGE XML format output
    output_pagexml_dir = os.path.join(output_dir, PAGE_XML_DIR)
    os.makedirs(output_pagexml_dir, exist_ok=True)

    # Store coordinates of page in a .txt file
    txt_coordinates = ''

    with tf.Session():  # Start a tensorflow session
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')

        for filename in tqdm(input_files, desc='Processed files'):
            # For each image, predict each pixel's label
            prediction_outputs = m.predict(filename)
            probs = prediction_outputs['probs'][0]
            original_shape = prediction_outputs['original_shape']
            probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
            probs = probs / np.max(probs)  # Normalize to be in [0, 1]

            # Binarize the predictions
            page_bin = page_make_binary_mask(probs)

            # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
            bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                      tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

            # Find quadrilateral enclosing the page
            pred_page_coords = boxes_detection.find_boxes(bin_upscaled.astype(np.uint8, copy=False),
                                                          mode='min_rectangle', n_max_boxes=1)

            # Draw page box on original image and export it. Add also box coordinates to the txt file
            original_img = imread(filename, mode='RGB')
            if pred_page_coords is not None:
                cv2.polylines(original_img, [pred_page_coords[:, None, :]], True, (0, 0, 255), thickness=5)
                # Write corners points into a .txt file
                txt_coordinates += '{},{}\n'.format(filename, format_quad_to_string(pred_page_coords))
            else:
                print('No box found in {}'.format(filename))
            basename = os.path.basename(filename).split('.')[0]
            imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), original_img)

            # Create page region and XML file
            page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(pred_page_coords[:, None, :]))
            page_xml = PAGE.Page(filename, image_width=original_shape[1], image_height=original_shape[0],
                                 page_border=page_border)
            xml_filename = os.path.join(output_pagexml_dir, '{}.xml'.format(basename))
            page_xml.write_to_file(xml_filename, creator_name='PageExtractor')

    # Save txt file
    with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        f.write(txt_coordinates)
