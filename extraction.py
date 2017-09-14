#!/usr/bin/env python
__author__ = 'solivr'

import tensorflow as tf
from doc_seg import loader
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import os
import argparse
import json
from scipy import misc


# gpu_to_use = '1'
# model_dir = 'data/models/posneg100_narrow_ornaments/vgg16_pretrained_conv2_w5_l5/export/'
# input_folder = 'data/reserve_ouvrages/images_for_prediciton/'
# output_folder = '/mnt/cluster-nas/sofia/BnF_export/'

def process_image(path_image, params_processing):
    img = Image.open(path_image)
    mat = np.asarray(img)
    mat_downres = np.asarray(img.resize(params_processing['resize_size']))
    if len(mat.shape) == 2:
        mat = np.stack([mat, mat, mat], axis=2)
        mat_downres = np.stack([mat_downres, mat_downres, mat_downres], axis=2)

    predictions_downres = m.predict(mat_downres[None], prediction_key='labels')[0]

    # This is for predictions that have no detected label (TODO correct classes.txt file)
    if np.count_nonzero(predictions_downres) - predictions_downres.size == 0:
        return

    if params_processing['upscaling']:
        predictions = misc.imresize(predictions_downres, mat.shape[:2], interp='nearest')
    else:
        predictions = predictions_downres

    # In this case the classes are 0 -> illustration, 1 -> background
    # We need to invert it (and TODO correct the order of the classes during training in the classes.txt file)
    prediction_formatted = 255*np.uint8(predictions == False)

    # Start by doing basing morphological cleaning
    kernel_open = np.ones(params_processing['kernelsize_opening'], np.uint8)
    opening = cv2.morphologyEx(prediction_formatted, cv2.MORPH_OPEN, kernel_open)

    # Then find the contours
    img_cnt, contours, h = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_ratio = 1.0/10.0
    crops_exported = False
    if contours:
        img_bgr = mat.copy()  # Change RGB to BGR for opencv
        img_bgr[:, :, 0] = mat[:, :, 2].copy()
        img_bgr[:, :, 2] = mat[:, :, 0].copy()
        img_with_boxes = img_bgr.copy()
        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            dimensions = sorted([h, w])
            if dimensions[0]/dimensions[1] < min_ratio:
                continue

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # TODO Ideally rotate the minAreaRect to have it horizontal and crop it
            crop = img_bgr[y:y+h, x:x+w, :]
            # Export crops
            file, extension = os.path.splitext(os.path.basename(path_image))
            filename_crop = os.path.join(params_processing['output_dir_crops'],
                                         '{}-{}{}'.format(file, i, extension))
            cv2.imwrite(filename_crop, crop)

            # Draw boxes
            img_with_boxes = cv2.drawContours(img_with_boxes, [box], 0, (255, 0, 0), 2)
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 8)
            crops_exported = True

        # Export image with boxes
        if crops_exported:
            filename_img_box = os.path.join(params_processing['output_dir_pages'],
                                            os.path.basename(path_image))
            cv2.imwrite(filename_img_box, img_with_boxes)
            if params_processing['debug']:
                filename_morph = os.path.join(params_processing['output_dir_pages'],
                                              '{}-morph.jpg'.format(os.path.splitext(os.path.basename(path_image))[0]))
                cv2.imwrite(filename_morph, opening)
    else:
        pass

# Make a function that checks whether a rectangle is inside a bigger rectangle
# def is_contained_in(r1, r2):
#     return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-dir", required=True, help="Folder with input images")
    ap.add_argument("-o", "--output-dir", required=True, help="Folder with output results")
    ap.add_argument("-m", "--model-dir", required=True, help="Where the model will be loaded")
    ap.add_argument("-g", "--gpu", type=str, required=True, help="Which GPU to use (0, 1)")
    args = vars(ap.parse_args())

    params_processing = {
        # 'resize_size': (416, 608),
        'resize_size': (480, 320),
        'kernelsize_opening': (20, 20),
        'upscaling': True,
        'output_dir_crops': os.path.join(args.get('output_dir'), 'crops'),
        'output_dir_pages': os.path.join(args.get('output_dir'), 'pages_with_boxes'),
        'input_dir': args.get('input_dir'),
        'model_dir': args.get('model_dir'),
        'debug': True
    }

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                                visible_device_list=args.get('gpu'))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
        m = loader.LoadedModel(args.get('model_dir'))

    input_images_filenames = glob(os.path.join(args.get('input_dir'), '**', '*.jpg'), recursive=True) + \
                             glob(os.path.join(args.get('input_dir'), '**', '*.png'), recursive=True)
    print('Found {} images'.format(len(input_images_filenames)))

    os.makedirs(params_processing['output_dir_crops'], exist_ok=True)
    os.makedirs(params_processing['output_dir_pages'], exist_ok=True)

    for path in tqdm(input_images_filenames):
        process_image(path, params_processing)

    # Exporting params
    with open(os.path.join(args.get('output_dir'), 'args.json'), 'w') as f:
        json.dump(args, f)
