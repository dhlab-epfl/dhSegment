#!/usr/bin/env python
__author__ = 'solivr'

from scipy.misc import imread, imresize
from scipy.ndimage import label
import tensorflow as tf
from doc_seg_datasets import post_processing, PAGE
from doc_seg.loader import LoadedModel
import numpy as np
import cv2
import xml.etree.ElementTree as ET


IMAGE_FILENAME = '/mnt/Garzoni/Garzoni/Giustizia-vecchia_Accordi-dei-garzoni-b-125-reg-179/Giustizia-vecchia_Accordi-dei-garzoni-b-125-reg-179_0014_007-v.tif'
IMAGE_HEIGHT = 1500
MODEL_DIR = '/scratch/sofia/tf_models/baseline_center/export/'
OUTPUT_XML_FILENAME = './page.xml'

img = imread(IMAGE_FILENAME)
img_resize = imresize(img, (IMAGE_HEIGHT, round(IMAGE_HEIGHT * (img.shape[1]/img.shape[0]))))
ratio_resizing = (img.shape[0]/img_resize.shape[0], img.shape[1]/img_resize.shape[1])

with tf.Session():
    model = LoadedModel('/scratch/sofia/tf_models/baseline_center/export/')
    predictions = model.predict(img_resize[None])

mask_predictions = post_processing.hysteresis_thresholding(probs=predictions['probs'][0, :, :, 1],
                                                           candidates=predictions['labels'][0, :, :],
                                                           low_threshold=0.2,
                                                           high_threshold=0.7)

kernel = np.zeros([7, 7], dtype=np.uint8)
kernel[2:5, :] = 1
mask_predictions_cleaned = cv2.morphologyEx((~mask_predictions).astype('uint8'), cv2.MORPH_CLOSE, kernel)
predictions_labeled, n_labels = label(mask_predictions_cleaned, structure=np.ones([3, 3]))

# Get TextLines :
# kernel = np.zeros([43, 43], dtype=np.uint8)
# kernel[:, 15:-15] = 1
# kernel[:5, :] = 0
k_size = int(IMAGE_HEIGHT*0.025)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
txt_lines_cnt = list()
for indice in range(1, n_labels - 1):
    element = cv2.morphologyEx((predictions_labeled * (predictions_labeled == indice)).astype('uint8'),
                               cv2.MORPH_DILATE, kernel)
    _, cnt_text_line, _ = cv2.findContours(element.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    txt_lines_cnt.append(post_processing.upscale_coordinates(cnt_text_line[0], ratio_resizing))


# Get text region : biggest enclosing rectangle
# 1. find all contours
_, cnt_text_region, _ = cv2.findContours(mask_predictions_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 2. Create a set of all the points
set_cnt = np.array([p for c in cnt_text_region for p in c[:, 0, :]])[:, None, :]
# 3. Get bounding rectangle
x, y, w, h = cv2.boundingRect(set_cnt)
x, y, w, h = int(round(x * ratio_resizing[1])), int(round(y * ratio_resizing[1])), \
             int(round(w * ratio_resizing[0])), int(round(h * ratio_resizing[1]))


xml = PAGE.create_xml_page({'image_filename': IMAGE_FILENAME,
                            'image_width': img.shape[1],
                            'image_height': img.shape[0],
                            'text_regions': [PAGE.TextRegion.from_array(coordinates=(x, y, w, h),
                                                                        text_lines=txt_lines_cnt)]
                            })

with open(OUTPUT_XML_FILENAME, 'w') as f:
    f.write(ET.tostring(xml, encoding='utf8', method='xml').decode('utf8'))