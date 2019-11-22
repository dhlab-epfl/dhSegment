#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
from typing import List
import os
import cv2
from imageio import imread, imsave
import numpy as np
import click
from tqdm import tqdm
from dh_segment.inference import LoadedModel
from process import page_post_processing_fn
from dh_segment.post_processing.boxes_detection import find_boxes
from dh_segment.io import PAGE

@click.command()
@click.argument('filenames_to_process', nargs=-1)
@click.option('--model_dir', help="The directory of te model to use")
@click.option('--output_dir', help="Directory to output the PAGEXML files")
@click.option('--draw_extractions', help="If true, the extracted lines will be drawn and exported to the output_dir")
def page_extraction(model_dir: str,
                    filenames_to_process: List[str],
                    output_dir: str,
                    draw_extractions: bool=False,
                    config: tf.ConfigProto=None):

    os.makedirs(output_dir, exist_ok=True)
    if draw_extractions:
        drawing_dir = os.path.join(output_dir, 'drawings')
        os.makedirs(drawing_dir)

    with tf.Session(config=config):
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename')
        for filename in tqdm(filenames_to_process, desc='Prediction'):
            # Inference
            prediction = m.predict(filename)
            probs = prediction['probs'][0]
            original_shape = prediction['original_shape']

            probs = probs / np.max(probs)  # Normalize to be in [0, 1]
            # Binarize the predictions
            page_bin = page_post_processing_fn(probs, threshold=-1)

            # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
            bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                      tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

            # Find quadrilateral enclosing the page
            pred_page_coords = find_boxes(bin_upscaled.astype(np.uint8, copy=False),
                                          mode='min_rectangle', n_max_boxes=1)

            if pred_page_coords is not None:
                # Write corners points into a .txt file

                # Create page region and XML file
                page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(pred_page_coords[:, None, :]))

                if draw_extractions:
                    # Draw page box on original image and export it. Add also box coordinates to the txt file
                    original_img = imread(filename, pilmode='RGB')
                    cv2.polylines(original_img, [pred_page_coords[:, None, :]], True, (0, 0, 255), thickness=5)

                    basename = os.path.basename(filename).split('.')[0]
                    imsave(os.path.join(drawing_dir, '{}_boxes.jpg'.format(basename)), original_img)

            else:
                print('No box found in {}'.format(filename))
                page_border = PAGE.Border()

            page_xml = PAGE.Page(image_filename=filename, image_width=original_shape[1], image_height=original_shape[0],
                                 page_border=page_border)
            xml_filename = os.path.join(output_dir, '{}.xml'.format(basename))
            page_xml.write_to_file(xml_filename, creator_name='PageExtractor')


if __name__ == '__main__':
    page_extraction()
