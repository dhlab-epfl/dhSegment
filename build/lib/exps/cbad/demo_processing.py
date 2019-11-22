#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import tensorflow as tf
import os
from typing import List
from dh_segment.inference import LoadedModel
from dh_segment.io import PAGE
from process import line_extraction_v1
from imageio import imread, imsave
from tqdm import tqdm
import click


@click.command()
@click.argument('filenames_to_process', nargs=-1)
@click.option('--model_dir', help="The directory of te model to use")
@click.option('--output_dir', help="Directory to output the PAGEXML files")
@click.option('--draw_extractions', help="If true, the extracted lines will be drawn and exported to the output_dir")
def baseline_extraction(model_dir: str,
                        filenames_to_process: List[str],
                        output_dir: str,
                        draw_extractions: bool=False,
                        config: tf.ConfigProto=None) -> None:
    """
    Given a model directory this function will load the model and apply it to the given files.

    :param model_dir: Directory containing the saved model
    :param filenames_to_process: filenames of the images to process
    :param output_dir: output directory to save the predictions (probability images)
    :param draw_extractions:
    :param config: ``ConfigProto`` object for ``tf.Session``.
    :return:
    """

    os.makedirs(output_dir, exist_ok=True)
    if draw_extractions:
        drawing_dir = os.path.join(output_dir, 'drawings')
        os.makedirs(drawing_dir)

    with tf.Session(config=config):
        # Load the model
        m = LoadedModel(model_dir, predict_mode='filename_original_shape')
        for filename in tqdm(filenames_to_process, desc='Prediction'):
            # Inference
            prediction = m.predict(filename)
            # Take the first element of the 'probs' dictionary (batch size = 1)
            probs = prediction['probs'][0]
            original_shape = probs.shape

            # The baselines probs are on the second channel
            baseline_probs = probs[:, :, 1]
            contours, _ = line_extraction_v1(baseline_probs, low_threshold=0.2, high_threshold=0.4, sigma=1.5)

            basename = os.path.basename(filename).split('.')[0]

            # Compute the ratio to save the coordinates in the original image coordinates reference.
            ratio = (original_shape[0] / probs.shape[0], original_shape[1] / probs.shape[1])
            xml_filename = os.path.join(output_dir, basename + '.xml')
            page_object = PAGE.save_baselines(xml_filename, contours, ratio, predictions_shape=probs.shape[:2])

            # If specified, saves the images with the annotated baslines
            if draw_extractions:
                image = imread(filename)
                page_object.draw_baselines(image, color=(255, 0, 0), thickness=5)

                basename = os.path.basename(filename)
                imsave(os.path.join(drawing_dir, basename), image)

if __name__ == '__main__':
    baseline_extraction()
