from glob import glob
import os
import tensorflow as tf
import numpy as np
from . import utils
from tqdm import tqdm
from random import shuffle
from .input_utils import data_augmentation_fn, extract_patches_fn, load_and_resize_image, rotate_crop


def input_fn(input_image_dir_or_filenames, params: dict, input_label_dir=None, data_augmentation=False,
             batch_size=5, make_patches=False, num_epochs=None, num_threads=4, image_summaries=False):

    training_params = utils.TrainingParams.from_dict(params['training_params'])
    prediction_type = params['prediction_type']
    classes_file = params['classes_file']

    # Finding the list of images to be used
    if isinstance(input_image_dir_or_filenames, list):
        input_images = input_image_dir_or_filenames
    else:
        input_images = glob(os.path.join(input_image_dir_or_filenames, '**', '*.jpg'), recursive=True) + \
                       glob(os.path.join(input_image_dir_or_filenames, '**', '*.png'), recursive=True)
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
            dataset = dataset.map(lambda filename: {'images':
                                                        load_and_resize_image(filename, 3,
                                                                              training_params.input_resized_size)})
        else:
            # Filenames
            encoded_filenames = [(i.encode(), l.encode()) for i, l in zip(input_images, label_images)]
            shuffle(encoded_filenames)
            dataset = tf.data.Dataset.from_generator(lambda: tqdm(encoded_filenames),
                                                     (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([])))
            dataset = dataset.repeat(count=num_epochs)

            # Load and resize images
            def _map_fn_1(image_filename, label_filename):
                if training_params.data_augmentation and training_params.input_resized_size > 0:
                    random_scaling = tf.random_uniform([],
                                                       np.maximum(1 - training_params.data_augmentation_max_scaling, 0),
                                                       1 + training_params.data_augmentation_max_scaling)
                    new_size = training_params.input_resized_size * random_scaling
                else:
                    new_size = training_params.input_resized_size

                if prediction_type in [utils.PredictionType.CLASSIFICATION, utils.PredictionType.MULTILABEL]:
                    label_image = load_and_resize_image(label_filename, 3, new_size, interpolation='NEAREST')
                elif prediction_type == utils.PredictionType.REGRESSION:
                    label_image = load_and_resize_image(label_filename, 1, new_size, interpolation='NEAREST')
                else:
                    raise NotImplementedError
                input_image = load_and_resize_image(image_filename, 3, new_size)
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

                if make_patches:
                    # Offsets for patch extraction
                    offsets = (tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32),
                               tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32))
                    # offsets = (0, 0)
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
                                                                                            training_params.data_augmentation_flip_ud,
                                                                                            training_params.data_augmentation_color))

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
        if make_patches:
            dataset = dataset.shuffle(50)

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


def serving_input_filename(resized_size):

    def serving_input_fn():
        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)

        # TODO : make it batch-compatible (with Dataset or string input producer)
        image = load_and_resize_image(filename, 3, resized_size)

        features = {'images': image[None]}

        receiver_inputs = {'filenames': filename}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs)

    return serving_input_fn


def serving_input_image():
    dic_input_serving = {'images': tf.placeholder(tf.float32, [None, None, None, 3])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(dic_input_serving)

