from glob import glob
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image import rotate as tf_rotate
from . import utils
from tqdm import tqdm
from random import shuffle


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

    def load_and_resize_image(filename, channels, interpolation='BILINEAR'):
        with tf.name_scope('load_img'):
            decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=channels,
                                                             try_recover_truncated=True))
            if training_params.input_resized_size > 0:
                with tf.name_scope('ImageRescaling'):
                    input_shape = tf.cast(tf.shape(decoded_image)[:2], tf.float32)
                    ratio_size = tf.div(training_params.input_resized_size,
                                        tf.reduce_prod(input_shape))
                    new_shape = tf.cast(tf.round(tf.multiply(ratio_size, input_shape)), tf.int32)
                    resize_method = {
                        'NEAREST': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        'BILINEAR': tf.image.ResizeMethod.BILINEAR
                    }
                    decoded_image = tf.image.resize_images(decoded_image, new_shape,
                                                           method=resize_method[interpolation])
            return decoded_image

    def make_patches_fn(input_image: tf.Tensor, label_image: tf.Tensor, offsets: tuple) -> (tf.Tensor, tf.Tensor):
        with tf.name_scope('patching'):
            patches_image = extract_patches_fn(input_image, training_params.patch_shape, offsets)
            patches_label = extract_patches_fn(label_image, training_params.patch_shape, offsets)

            return patches_image, patches_label

    # Tensorflow input_fn
    def fn():
        if not input_label_dir:
            encoded_filenames = [f.encode() for f in input_images]
            dataset = tf.data.Dataset.from_generator(lambda: tqdm(encoded_filenames),
                                                     tf.string, tf.TensorShape([]))
            dataset = dataset.repeat(count=num_epochs)
            dataset = dataset.map(lambda filename: {'images': load_and_resize_image(filename, 3)})
        else:
            # Filenames
            encoded_filenames = [(i.encode(), l.encode()) for i, l in zip(input_images, label_images)]
            shuffle(encoded_filenames)
            dataset = tf.data.Dataset.from_generator(lambda: tqdm(encoded_filenames),
                                                     (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([])))

            # Load and resize images
            def _map_fn_1(image_filename, label_filename):
                if prediction_type in [utils.PredictionType.CLASSIFICATION, utils.PredictionType.MULTILABEL]:
                    label_image = load_and_resize_image(label_filename, 3, interpolation='NEAREST')
                elif prediction_type == utils.PredictionType.REGRESSION:
                    label_image = load_and_resize_image(label_filename, 1, interpolation='NEAREST')
                else:
                    raise NotImplementedError
                input_image = load_and_resize_image(image_filename, 3)
                return input_image, label_image

            dataset = dataset.map(_map_fn_1, num_threads)

            # Data augmentation, patching
            def _map_fn_2(input_image, label_image):
                if data_augmentation:
                    # Rotation of the original image
                    if training_params.data_augmentation_max_rotation > 0:
                        with tf.name_scope('random_rotation'):
                            rotation_angle = tf.random_uniform([],
                                                               -training_params.data_augmentation_max_rotation,
                                                               training_params.data_augmentation_max_rotation)
                            label_image = rotate_crop(label_image, rotation_angle,
                                                      minimum_shape=[2 * i for i in training_params.patch_shape],
                                                      interpolation='NEAREST')
                            input_image = rotate_crop(input_image, rotation_angle,
                                                      minimum_shape=[2 * i for i in training_params.patch_shape],
                                                      interpolation='BILINEAR')

                    # Offsets for patch extraction
                    offsets = (tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32),
                               tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32))
                else:
                    offsets = (0, 0)

                if make_patches:
                    batch_image, batch_label = make_patches_fn(input_image, label_image, offsets)
                else:
                    with tf.name_scope('formatting'):
                        batch_image = tf.expand_dims(input_image, axis=0)
                        batch_label = tf.expand_dims(label_image, axis=0)
                return tf.data.Dataset.from_tensor_slices((batch_image, batch_label))

            dataset = dataset.flat_map(_map_fn_2)

            if data_augmentation:
                dataset = dataset.map(lambda input_image, label_image: data_augmentation_fn(input_image,
                                                                                            label_image,
                                                                                            training_params.data_augmentation_flip_lr,
                                                                                            training_params.data_augmentation_flip_ud))

            # Assign color to class id
            def _map_fn_3(input_image, label_image):
                # Convert RGB to class id
                if prediction_type == utils.PredictionType.CLASSIFICATION:
                    label_image = utils.label_image_to_class(label_image, classes_file)
                elif prediction_type == utils.PredictionType.MULTILABEL:
                    label_image = utils.multilabel_image_to_class(label_image, classes_file)
                return {'images': input_image, 'labels': label_image}

            dataset = dataset.map(_map_fn_3, num_threads)

        # Save original size of images
        dataset = dataset.map(lambda d: {'shapes': tf.shape(d['images'])[:2], **d})

        # Pad things
        padded_shapes = {
            'images': [-1, -1, 3],
            'shapes': [2]
        }
        if 'labels' in dataset.output_shapes.keys():
            output_shapes_label = dataset.output_shapes['labels']
            padded_shapes['labels'] = [-1, -1] + list(output_shapes_label[2:])
        dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)

        dataset = dataset.prefetch(4)
        dataset = dataset.repeat(count=num_epochs)
        iterator = dataset.make_one_shot_iterator()
        prepared_batch = iterator.get_next()

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            shape_summary_img = training_params.patch_shape if make_patches else training_params.input_resized_shape
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


def data_augmentation_fn(input_image: tf.Tensor, label_image: tf.Tensor,
                         flip_lr: bool=True, flip_ud: bool=True) -> (tf.Tensor, tf.Tensor):
    with tf.name_scope('DataAugmentation'):
        if flip_lr:
            with tf.name_scope('random_flip_lr'):
                sample = tf.random_uniform([], 0, 1)
                label_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(label_image), lambda: label_image)
                input_image = tf.cond(sample > 0.5, lambda: tf.image.flip_left_right(input_image), lambda: input_image)
        if flip_ud:
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


def rotate_crop(img, rotation, crop=True, minimum_shape=[0, 0], interpolation='NEAREST'):
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
            rotated_image = tf.cond(tf.less_equal(tf.reduce_min(tf.shape(rotated_image_crop)[:2]),
                                                  tf.reduce_max(minimum_shape)),
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


def serving_input_filename():

    def serving_input_fn():
        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)

        # TODO : make it batch-compatible (with Dataset or string input producer)
        with tf.name_scope('load_img'):
            image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=3,
                                                     try_recover_truncated=True))

        features = {'images': image[None]}

        receiver_inputs = {'filenames': filename}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs)

    return serving_input_fn


def serving_input_image():
    dic_input_serving = {'images': tf.placeholder(tf.float32, [None, None, None, 3])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(dic_input_serving)


def prediction_input_filename(filename):

    def fn():
        with tf.name_scope('load_img'):
            image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=3,
                                                     try_recover_truncated=True))
        features = {'images': image[None]}
        return features

    return fn
