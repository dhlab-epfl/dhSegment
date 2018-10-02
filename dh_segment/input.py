from glob import glob
import os
import tensorflow as tf
import numpy as np
from . import utils
from tqdm import tqdm
from typing import Union, List
from enum import Enum
import pandas as pd
from .input_utils import data_augmentation_fn, extract_patches_fn, load_and_resize_image, \
    rotate_crop, resize_image, local_entropy


class InputCase(Enum):
    INPUT_LIST = 'INPUT_LIST'
    INPUT_DIR = 'INPUT_DIR'
    INPUT_CSV = 'INPUT_CSV'


def input_fn(input_data: Union[str, List[str]], params: dict, input_label_dir: str=None,
             data_augmentation: bool=False, batch_size: int=5, make_patches: bool=False, num_epochs: int=None,
             num_threads: int=4, image_summaries: bool=False):
    """
    Input_fn for estimator
    :param input_data: input data. It can be a directory containing the images, it can be
        a list of image filenames, or it can be a path to a csv file.
    :param params: params from utils.Params object
    :param input_label_dir: directory containing the label images
    :param data_augmentation: boolean, if True will scale, roatate, ... the images
    :param batch_size: size of the bach
    :param make_patches: bool, whether to make patches (crop image in smaller pieces) or not
    :param num_epochs: number of epochs to cycle trough data
    :param num_threads: number of thread to use in parallele when usin tf.data.Dataset.map
    :param image_summaries: boolean, whether to make tf.Summary to watch on tensorboard
    :return: fn
    """
    training_params = utils.TrainingParams.from_dict(params['training_params'])
    prediction_type = params['prediction_type']
    classes_file = params['classes_file']

    # --- Map functions
    def _make_patches_fn(input_image: tf.Tensor, label_image: tf.Tensor, offsets: tuple) -> (tf.Tensor, tf.Tensor):
        with tf.name_scope('patching'):
            patches_image = extract_patches_fn(input_image, training_params.patch_shape, offsets)
            patches_label = extract_patches_fn(label_image, training_params.patch_shape, offsets)

            return patches_image, patches_label

    # Load and resize images
    def _load_image_fn(image_filename, label_filename):
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

    # Data augmentation, patching
    def _scaling_and_patch_fn(input_image, label_image):
        if data_augmentation:
            # Rotation of the original image
            if training_params.data_augmentation_max_rotation > 0:
                with tf.name_scope('random_rotation'):
                    rotation_angle = tf.random_uniform([],
                                                       -training_params.data_augmentation_max_rotation,
                                                       training_params.data_augmentation_max_rotation)
                    label_image = rotate_crop(label_image, rotation_angle,
                                              minimum_shape=[(i * 3) // 2 for i in training_params.patch_shape],
                                              interpolation='NEAREST')
                    input_image = rotate_crop(input_image, rotation_angle,
                                              minimum_shape=[(i * 3) // 2 for i in training_params.patch_shape],
                                              interpolation='BILINEAR')

        if make_patches:
            # Offsets for patch extraction
            offsets = (tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32),
                       tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32))
            # offsets = (0, 0)
            batch_image, batch_label = _make_patches_fn(input_image, label_image, offsets)
        else:
            with tf.name_scope('formatting'):
                batch_image = tf.expand_dims(input_image, axis=0)
                batch_label = tf.expand_dims(label_image, axis=0)
        return tf.data.Dataset.from_tensor_slices((batch_image, batch_label))

    # Data augmentation
    def _augment_data_fn(input_image, label_image): \
        return data_augmentation_fn(input_image, label_image, training_params.data_augmentation_flip_lr,
                                    training_params.data_augmentation_flip_ud, training_params.data_augmentation_color)

    # Assign color to class id
    def _assign_color_to_class_id(input_image, label_image):
        # Convert RGB to class id
        if prediction_type == utils.PredictionType.CLASSIFICATION:
            label_image = utils.label_image_to_class(label_image, classes_file)
        elif prediction_type == utils.PredictionType.MULTILABEL:
            label_image = utils.multilabel_image_to_class(label_image, classes_file)
        output = {'images': input_image, 'labels': label_image}

        if training_params.local_entropy_ratio > 0 and prediction_type == utils.PredictionType.CLASSIFICATION:
            output['weight_maps'] = local_entropy(tf.equal(label_image, 1),
                                                  sigma=training_params.local_entropy_sigma)
        return output
    # ---

    # Finding the list of images to be used
    if isinstance(input_data, list):
        input_case = InputCase.INPUT_LIST
        input_image_filenames = input_data
        print('Found {} images'.format(len(input_image_filenames)))

    elif os.path.isdir(input_data):
        input_case = InputCase.INPUT_DIR
        input_image_filenames = glob(os.path.join(input_data, '**', '*.jpg'),
                                     recursive=True) + \
                                glob(os.path.join(input_data, '**', '*.png'),
                                     recursive=True)
        print('Found {} images'.format(len(input_image_filenames)))

    elif os.path.isfile(input_data) and \
            input_data.endswith('.csv'):
        input_case = InputCase.INPUT_CSV
    else:
        raise NotImplementedError('Input data should be a directory, a csv file or a list of filenames but got {}'.format(input_data))

    # Finding the list of labelled images if available
    has_labelled_data = False
    if input_label_dir and input_case in [InputCase.INPUT_LIST, InputCase.INPUT_DIR]:
        label_image_filenames = []
        for input_image_filename in input_image_filenames:
            label_image_filename = os.path.join(input_label_dir, os.path.basename(input_image_filename))
            if not os.path.exists(label_image_filename):
                filename, extension = os.path.splitext(os.path.basename(input_image_filename))
                new_extension = '.png' if extension == '.jpg' else '.jpg'
                label_image_filename = os.path.join(input_label_dir, filename + new_extension)
            label_image_filenames.append(label_image_filename)
        has_labelled_data = True

    # Read image filenames and labels in case of csv file
    if input_case == InputCase.INPUT_CSV:
        df = pd.read_csv(input_data, header=None, names=['images', 'labels'])
        input_image_filenames = list(df.images.values)
        # If the label column exists
        if not np.alltrue(pd.isnull(df.labels.values)):
            label_image_filenames = list(df.labels.values)
            has_labelled_data = True

    # Checks that all image files can be found
    for img_filename in input_image_filenames:
        if not os.path.exists(img_filename):
            raise FileNotFoundError(img_filename)
    if has_labelled_data:
        for img_filename in input_image_filenames:
            if not os.path.exists(img_filename):
                raise FileNotFoundError(img_filename)

    # Tensorflow input_fn
    def fn():
        if not has_labelled_data:
            encoded_filenames = [f.encode() for f in input_image_filenames]
            dataset = tf.data.Dataset.from_generator(lambda: tqdm(encoded_filenames, desc='Dataset'),
                                                     tf.string, tf.TensorShape([]))
            dataset = dataset.repeat(count=num_epochs)
            dataset = dataset.map(lambda filename: {'images': load_and_resize_image(filename, 3,
                                                                                    training_params.input_resized_size)})
        else:
            encoded_filenames = [(i.encode(), l.encode()) for i, l in zip(input_image_filenames, label_image_filenames)]
            dataset = tf.data.Dataset.from_generator(lambda: tqdm(utils.shuffled(encoded_filenames), desc='Dataset'),
                                                         (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([])))

            dataset = dataset.repeat(count=num_epochs)
            dataset = dataset.map(_load_image_fn, num_threads).flat_map(_scaling_and_patch_fn)

            if data_augmentation:
                dataset = dataset.map(_augment_data_fn, num_threads)
            dataset = dataset.map(_assign_color_to_class_id, num_threads)

        # Save original size of images
        dataset = dataset.map(lambda d: {'shapes': tf.shape(d['images'])[:2], **d})
        if make_patches:
            dataset = dataset.shuffle(128)

        if make_patches and input_label_dir:
            base_shape_images = list(training_params.patch_shape)
        else:
            base_shape_images = [-1, -1]
        # Pad things
        padded_shapes = {
            'images': base_shape_images + [3],
            'shapes': [2]
        }
        if 'labels' in dataset.output_shapes.keys():
            output_shapes_label = dataset.output_shapes['labels']
            padded_shapes['labels'] = base_shape_images + list(output_shapes_label[2:])
        if 'weight_maps' in dataset.output_shapes.keys():
            padded_shapes['weight_maps'] = base_shape_images

        dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes).prefetch(8)
        prepared_batch = dataset.make_one_shot_iterator().get_next()

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            shape_summary_img = tf.cast(tf.shape(prepared_batch['images'])[1:3] / 3, tf.int32)
            tf.summary.image('input/image',
                             tf.image.resize_images(prepared_batch['images'], shape_summary_img),
                             max_outputs=1)
            if 'labels' in prepared_batch:
                label_export = prepared_batch['labels']
                if prediction_type == utils.PredictionType.CLASSIFICATION:
                    label_export = utils.class_to_label_image(label_export, classes_file)
                if prediction_type == utils.PredictionType.MULTILABEL:
                    label_export = tf.cast(label_export, tf.int32)
                    label_export = utils.multiclass_to_label_image(label_export, classes_file)
                tf.summary.image('input/label',
                                 tf.image.resize_images(label_export, shape_summary_img), max_outputs=1)
            if 'weight_maps' in prepared_batch:
                tf.summary.image('input/weight_map',
                                 tf.image.resize_images(prepared_batch['weight_maps'][:, :, :, None],
                                                        shape_summary_img),
                                 max_outputs=1)

        return prepared_batch, prepared_batch.get('labels')

    return fn


def serving_input_filename(resized_size):
    def serving_input_fn():
        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)

        # TODO : make it batch-compatible (with Dataset or string input producer)
        decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=3,
                                                         try_recover_truncated=True))
        original_shape = tf.shape(decoded_image)[:2]

        if resized_size is not None and resized_size > 0:
            image = resize_image(decoded_image, resized_size)
        else:
            image = decoded_image

        image_batch = image[None]
        features = {'images': image_batch, 'original_shape': original_shape}

        receiver_inputs = {'filename': filename}

        input_from_resized_images = {'resized_images': image_batch}
        input_from_original_image = {'image': decoded_image}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs,
                                                        receiver_tensors_alternatives={'from_image':
                                                                                           input_from_original_image,
                                                                                       'from_resized_images':
                                                                                           input_from_resized_images})

    return serving_input_fn


def serving_input_image():
    dic_input_serving = {'images': tf.placeholder(tf.float32, [None, None, None, 3])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(dic_input_serving)
