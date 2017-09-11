from glob import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image import rotate as tf_rotate
from . import utils


def input_fn(prediction_type: utils.PredictionType, input_folder, label_images_folder=None, classes_file=None,
             data_augmentation=False, resized_size=(600, 400), make_patches=False, batch_size=5, num_epochs=None,
             num_threads=4, image_summaries=False):
    # Finding the list of images to be used
    input_images = glob(os.path.join(input_folder, '**', '*.jpg'), recursive=True) + \
                   glob(os.path.join(input_folder, '**', '*.png'), recursive=True)
    print('Found {} images'.format(len(input_images)))

    # Finding the list of labelled images if available
    if label_images_folder:
        if prediction_type == utils.PredictionType.CLASSIFICATION:
            assert classes_file is not None, "Needs a classes_file if training"
        label_images = []
        for input_image_filename in input_images:
            label_image_filename = os.path.join(label_images_folder, os.path.basename(input_image_filename))
            if not os.path.exists(label_image_filename):
                filename, extension = os.path.splitext(os.path.basename(input_image_filename))
                new_extension = '.png' if extension == '.jpg' else '.jpg'
                label_image_filename = os.path.join(label_images_folder, filename + new_extension)
                if not os.path.exists(label_image_filename):
                    raise FileNotFoundError(label_image_filename)

            label_images.append(label_image_filename)

    # Helper loading function
    def load_image(filename, channels):
        with tf.name_scope('load_resize_img'):
            decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=channels,
                                                             try_recover_truncated=True))

            return decoded_image

    # Tensorflow input_fn
    def fn():
        if not label_images_folder:
            image_filename = tf.train.string_input_producer(input_images, num_epochs=num_epochs).dequeue()
            to_batch = {'images': tf.image.resize_images(load_image(image_filename, 3), resized_size)}
        else:
            # Get one filename of each
            image_filename, label_filename = tf.train.slice_input_producer([input_images, label_images],
                                                                           num_epochs=num_epochs,
                                                                           shuffle=True)
            # Read and resize the images
            if prediction_type == utils.PredictionType.CLASSIFICATION:
                label_image = load_image(label_filename, 3)
            elif prediction_type == utils.PredictionType.REGRESSION:
                label_image = load_image(label_filename, 1)
            else:
                raise NotImplementedError
            input_image = load_image(image_filename, 3)
            # Parallel data augmentation
            if data_augmentation:
                input_image, label_image = data_augmentation_fn(input_image, label_image)

            if make_patches:
                patch_shape = resized_size
                if data_augmentation:
                    offsets = (tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32),
                               tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32))
                else:
                    offsets = (0, 0)
                patches_image = extract_patches_fn(input_image, patch_shape, offsets)
                patches_label = extract_patches_fn(label_image, patch_shape, offsets)
            else:
                with tf.name_scope('formatting'):
                    patches_image = tf.expand_dims(tf.image.resize_images(input_image, resized_size), axis=0)
                    patches_label = tf.expand_dims(tf.image.resize_images(label_image, resized_size), axis=0)

            # #  see https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
            # input_image = tf.image.resize_images(input_image, resized_size)
            # label_image = tf.image.resize_images(label_image, resized_size,
            #                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if prediction_type == utils.PredictionType.CLASSIFICATION:
                # Convert RGB to class id
                patches_label = utils.label_image_to_class(patches_label, classes_file)
            to_batch = {'images': patches_image, 'labels': patches_label}

        # Batch the preprocessed images
        min_after_dequeue = 50
        prepared_batch = tf.train.shuffle_batch(to_batch, batch_size=batch_size, num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue,
                                                capacity=max(3 * batch_size * num_threads, 1.5*min_after_dequeue),
                                                allow_smaller_final_batch=True, enqueue_many=True)

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            tf.summary.image('input/image',
                             tf.image.resize_images(prepared_batch['images'], np.array(resized_size) / 3),
                             max_outputs=1)
            if 'labels' in prepared_batch:
                label_export = prepared_batch['labels']
                if prediction_type == utils.PredictionType.CLASSIFICATION:
                    label_export = utils.class_to_label_image(label_export, classes_file)
                tf.summary.image('input/label',
                                 tf.image.resize_images(label_export, np.array(resized_size) / 3), max_outputs=1)

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
        with tf.name_scope('random_rotate'):
            rotation_angle = tf.random_uniform([], -0.05, 0.05)
            label_image = rotate_crop(label_image, rotation_angle, interpolation='NEAREST')
            input_image = rotate_crop(input_image, rotation_angle, interpolation='BILINEAR')

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