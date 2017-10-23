from glob import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image import rotate as tf_rotate
from tensorflow.python.training import queue_runner
from . import utils


def input_fn(input_image_dir, params: dict, input_label_dir=None, data_augmentation=False,
             batch_size=5, make_patches=False, num_epochs=None, num_threads=4, image_summaries=False):

    training_params = utils.TrainingParams.from_dict(params['training_params'])
    prediction_type = params['prediction_type']
    classes_file = params['classes_file']

    # Finding the list of images to be used
    input_images = glob(os.path.join(input_image_dir, '**', '*.jpg'), recursive=True) + \
                   glob(os.path.join(input_image_dir, '**', '*.png'), recursive=True)
    print('Found {} images'.format(len(input_images)))

    # Finding the list of labelled images if available
    if input_label_dir:
        label_images = []
        for input_image_filename in input_images:
            label_image_filename = os.path.join(input_label_dir, os.path.basename(input_image_filename))
            if not os.path.exists(label_image_filename):
                filename, extension = os.path.splitext(os.path.basename(input_image_filename))
                new_extension = '.png' if extension == '.jpg' else '.jpg'
                label_image_filename = os.path.join(input_label_dir, filename + new_extension)
                if not os.path.exists(label_image_filename):
                    raise FileNotFoundError(label_image_filename)

            label_images.append(label_image_filename)

    def load_image(filename, channels):
        with tf.name_scope('load_img'):
            decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=channels,
                                                             try_recover_truncated=True))

            return decoded_image

    def make_patches_fn(input_image: tf.Tensor, label_image: tf.Tensor, offsets: tuple) -> (tf.Tensor, tf.Tensor):
        with tf.name_scope('patching'):
            patches_image = extract_patches_fn(input_image, training_params.patch_shape, offsets)
            patches_label = extract_patches_fn(label_image, training_params.patch_shape, offsets)

            with tf.name_scope('patches_queue'):
                # Data augmentation directly on patches
                queue_patches = tf.FIFOQueue(capacity=3000, dtypes=[tf.float32, tf.float32])
                # Enqueue all
                enqueue_op = queue_patches.enqueue_many((patches_image, patches_label))
                queue_runner.add_queue_runner(tf.train.QueueRunner(queue_patches, [enqueue_op] * 2))
                # Dequeue one by one
                dequeue_patch_img, dequeue_patch_lab = queue_patches.dequeue()

                dequeue_patch_img.set_shape([*training_params.patch_shape, 3])
                dequeue_patch_lab.set_shape([*training_params.patch_shape, 1])

            patches_image = dequeue_patch_img
            patches_label = dequeue_patch_lab

            return patches_image, patches_label

    # Tensorflow input_fn
    def fn():
        if not input_label_dir:
            image_filename = tf.train.string_input_producer(input_images, num_epochs=num_epochs).dequeue()
            to_batch = {'images': tf.image.resize_images(load_image(image_filename, 3),
                                                         training_params.input_resized_size)}
        else:
            # Get one filename of each
            image_filename, label_filename = tf.train.slice_input_producer([input_images, label_images],
                                                                           num_epochs=num_epochs,
                                                                           shuffle=True)
            # Load images
            if prediction_type == utils.PredictionType.CLASSIFICATION or \
                            prediction_type == utils.PredictionType.MULTILABEL:
                label_image = load_image(label_filename, 3)
            elif prediction_type == utils.PredictionType.REGRESSION:
                label_image = load_image(label_filename, 1)

            input_image = load_image(image_filename, 3)

            if data_augmentation:
                # Rotation of the original image
                with tf.name_scope('random_rotation'):
                    rotation_angle = tf.random_uniform([], -0.1, 0.1)
                    label_image = rotate_crop(label_image, rotation_angle, interpolation='NEAREST')
                    input_image = rotate_crop(input_image, rotation_angle, interpolation='BILINEAR')

                # Offsets for patch extraction
                offsets = (tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32),
                           tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32))
            else:
                offsets = (0, 0)

            if make_patches:
                formatted_image, formatted_label = make_patches_fn(input_image, label_image, offsets)
            else:
                with tf.name_scope('formatting'):
                    with tf.name_scope('resizing'):
                        input_image = tf.image.resize_images(input_image, training_params.input_resized_size,
                                                             method=tf.image.ResizeMethod.BILINEAR)
                        label_image = tf.image.resize_images(label_image, training_params.input_resized_size,
                                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                    formatted_image = input_image
                    formatted_label = label_image

            if data_augmentation:
                formatted_image, formatted_label = data_augmentation_fn(formatted_image, formatted_label)

            with tf.name_scope('dim_expansion'):
                batch_image = tf.expand_dims(formatted_image, axis=0)
                batch_label = tf.expand_dims(formatted_label, axis=0)

            # Convert RGB to class id
            if prediction_type == utils.PredictionType.CLASSIFICATION:
                batch_label = utils.label_image_to_class(batch_label, classes_file)
            elif prediction_type == utils.PredictionType.MULTILABEL:
                batch_label = utils.multilabel_image_to_class(batch_label, classes_file)

            to_batch = {'images': batch_image, 'labels': batch_label}

        # Batch the preprocessed images
        min_after_dequeue = 50
        prepared_batch = tf.train.shuffle_batch(to_batch, batch_size=batch_size, num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue,
                                                capacity=max(3 * batch_size * num_threads, 1.5*min_after_dequeue),
                                                allow_smaller_final_batch=False,  # Keep to False
                                                enqueue_many=True)

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            shape_summary_img = training_params.patch_shape if make_patches else training_params.input_resized_size
            tf.summary.image('input/image',
                             tf.image.resize_images(prepared_batch['images'], np.array(shape_summary_img) / 3),
                             max_outputs=1)
            if 'labels' in prepared_batch:
                label_export = prepared_batch['labels']
                if prediction_type == utils.PredictionType.CLASSIFICATION:
                    label_export = utils.class_to_label_image(label_export, classes_file)
                if prediction_type == utils.PredictionType.MULTILABEL:
                    label_export = tf.cast(label_export, tf.int32)
                    label_export.set_shape((batch_size, *shape_summary_img, None))
                    label_export = utils.multiclass_to_label_image(label_export, classes_file)
                    tf.summary.image('input/label',
                                     tf.image.resize_images(label_export, np.array(shape_summary_img) / 3), max_outputs=1)

        return prepared_batch, prepared_batch.get('labels')

    return fn


def data_augmentation_fn(input_image: tf.Tensor, label_image: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    with tf.name_scope('DataAugmentation'):
        with tf.name_scope('random_flip_lr'):
            sample = tf.random_uniform([], 0, 1)
            label_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(label_image), lambda: label_image)
            input_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
        with tf.name_scope('random_flip_ud'):
            sample = tf.random_uniform([], 0, 1)
            label_image = tf.cond(sample > 0.5, lambda: tf.image.flip_up_down(label_image), lambda: label_image)
            input_image = tf.cond(sample > 0.5, lambda: tf.image.flip_up_down(input_image), lambda: input_image)

        chanels = input_image.get_shape()[-1]
        input_image = tf.image.random_contrast(input_image, lower=0.8, upper=1.0)
        if chanels == 3:
            input_image = tf.image.random_hue(input_image, max_delta=0.1)
            input_image = tf.image.random_saturation(input_image, lower=0.8, upper=1.2)
        return input_image, label_image


def rotate_crop(img, rotation, crop=True, interpolation='NEAREST'):
    with tf.name_scope('RotateCrop'):
        rotated_image = tf_rotate(img, rotation, interpolation)
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            # see https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2 * rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h - new_h) / 2), tf.int32), tf.cast(tf.ceil((w - new_w) / 2), tf.int32)
            rotated_image_crop = rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :]

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop)
        return rotated_image


def extract_patches_fn(image: tf.Tensor, patch_shape: list, offsets) -> tf.Tensor:
    """
    :param image: tf.Tensor
    :param patch_shape: [h, w]
    :param offsets: tuple between 0 and 1
    :return: patches [batch_patches, h, w, c]
    """
    with tf.name_scope('patch_extraction'):
        h, w = patch_shape
        c = image.get_shape()[-1]

        offset_h = tf.cast(tf.round(offsets[0] * h // 2), dtype=tf.int32)
        offset_w = tf.cast(tf.round(offsets[1] * w // 2), dtype=tf.int32)
        offset_img = image[offset_h:, offset_w:, :]
        offset_img = offset_img[None, :, :, :]

        patches = tf.extract_image_patches(offset_img, ksizes=[1, h, w, 1], strides=[1, h // 2, w // 2, 1],
                                           rates=[1, 1, 1, 1], padding='VALID')
        patches_shape = tf.shape(patches)
        return tf.reshape(patches, [tf.reduce_prod(patches_shape[0:3]), h, w, int(c)])  # returns [batch_patches, h, w, c]