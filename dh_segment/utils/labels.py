#!/usr/bin/env python
__license__ = "GPL"

import tensorflow as tf
import numpy as np
import os


def label_image_to_class(label_image: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values = get_classes_color_from_file(classes_file)
    # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
    with tf.name_scope('LabelAssign'):
        if len(label_image.get_shape()) == 3:
            diff = tf.cast(label_image[:, :, None, :], tf.float32) - tf.constant(classes_color_values[None, None, :, :])  # [H,W,C,3]
        elif len(label_image.get_shape()) == 4:
            diff = tf.cast(label_image[:, :, :, None, :], tf.float32) - tf.constant(
                classes_color_values[None, None, None, :, :])  # [B,H,W,C,3]
        else:
            raise NotImplementedError('Length is : {}'.format(len(label_image.get_shape())))

        pixel_class_diff = tf.reduce_sum(tf.square(diff), axis=-1)  # [H,W,C] or [B,H,W,C]
        class_label = tf.argmin(pixel_class_diff, axis=-1)  # [H,W] or [B,H,W]
        return class_label


def class_to_label_image(class_label: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values = get_classes_color_from_file(classes_file)
    return tf.gather(classes_color_values, tf.cast(class_label, dtype=tf.int32))


def multilabel_image_to_class(label_image: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values, colors_labels = get_classes_color_from_file_multilabel(classes_file)
    # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
    with tf.name_scope('LabelAssign'):
        if len(label_image.get_shape()) == 3:
            diff = tf.cast(label_image[:, :, None, :], tf.float32) - tf.constant(classes_color_values[None, None, :, :])  # [H,W,C,3]
        elif len(label_image.get_shape()) == 4:
            diff = tf.cast(label_image[:, :, :, None, :], tf.float32) - tf.constant(
                classes_color_values[None, None, None, :, :])  # [B,H,W,C,3]
        else:
            raise NotImplementedError('Length is : {}'.format(len(label_image.get_shape())))

        pixel_class_diff = tf.reduce_sum(tf.square(diff), axis=-1)  # [H,W,C] or [B,H,W,C]
        class_label = tf.argmin(pixel_class_diff, axis=-1)  # [H,W] or [B,H,W]

        return tf.gather(colors_labels, class_label) > 0


def multiclass_to_label_image(class_label_tensor: tf.Tensor, classes_file: str) -> tf.Tensor:

    classes_color_values, colors_labels = get_classes_color_from_file_multilabel(classes_file)

    n_classes = colors_labels.shape[1]
    c = np.zeros((2,)*n_classes+(3,), np.int32)
    for c_value, inds in zip(classes_color_values, colors_labels):
        c[tuple(inds)] = c_value

    with tf.name_scope('Label2Img'):
        return tf.gather_nd(c, tf.cast(class_label_tensor, tf.int32))


def get_classes_color_from_file(classes_file: str) -> np.ndarray:
    if not os.path.exists(classes_file):
        raise FileNotFoundError(classes_file)
    result = np.loadtxt(classes_file).astype(np.float32)
    assert result.shape[1] == 3, "Color file should represent RGB values"
    return result


def get_n_classes_from_file(classes_file: str) -> int:
    return get_classes_color_from_file(classes_file).shape[0]


def get_classes_color_from_file_multilabel(classes_file: str) -> np.ndarray:
    if not os.path.exists(classes_file):
        raise FileNotFoundError(classes_file)
    result = np.loadtxt(classes_file).astype(np.float32)
    assert result.shape[1] > 3, "The number of columns should be greater in multilabel framework"
    colors = result[:, :3]
    labels = result[:, 3:]
    return colors, labels.astype(np.int32)


def get_n_classes_from_file_multilabel(classes_file: str) -> int:
    return get_classes_color_from_file_multilabel(classes_file)[1].shape[1]
