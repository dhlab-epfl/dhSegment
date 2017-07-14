import tensorflow as tf
import numpy as np
import os


class PredictionType:
    CLASSIFICATION = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'


def label_image_to_class(label_image: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values = _get_classes_color_from_file(classes_file)
    # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
    with tf.name_scope('LabelAssign'):
        if len(label_image.get_shape()) == 3:
            diff = tf.cast(label_image[:, :, None, :], tf.float32) - tf.constant(classes_color_values[None, None, :, :])  # [H,W,C,3]
        elif len(label_image.get_shape()) == 4:
            diff = tf.cast(label_image[:, :, :, None, :], tf.float32) - tf.constant(
                classes_color_values[None, None, None, :, :])  # [B,H,W,C,3]
        else:
            raise NotImplementedError

        pixel_class_diff = tf.reduce_sum(tf.square(diff), axis=-1)  # [H,W,C] or [B,H,W,C]
        class_label = tf.argmin(pixel_class_diff, axis=-1)  # [H,W] or [B,H,W]
        return class_label


def class_to_label_image(class_label: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values = _get_classes_color_from_file(classes_file)
    return tf.gather(classes_color_values, class_label)


def _get_classes_color_from_file(classes_file: str) -> np.ndarray:
    if not os.path.exists(classes_file):
        raise FileNotFoundError(classes_file)
    result = np.loadtxt(classes_file).astype(np.float32)
    assert result.shape[1] == 3, "Color file should represent RGB values"
    return result