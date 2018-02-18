from glob import glob
import os
import tensorflow as tf
import numpy as np
from . import utils
from tqdm import tqdm
from random import shuffle
from .input_utils import data_augmentation_fn, extract_patches_fn, load_and_resize_image, rotate_crop
from scipy import ndimage


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
                output = {'images': input_image, 'labels': label_image}

                if training_params.local_entropy_ratio > 0 and prediction_type == utils.PredictionType.CLASSIFICATION:
                    output['weight_maps'] = local_entropy(tf.equal(label_image, 1),
                                                          sigma=training_params.local_entropy_sigma)
                return output

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
        if 'weight_maps' in dataset.output_shapes.keys():
            padded_shapes['weight_maps'] = [-1, -1]
        dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)
        dataset = dataset.prefetch(4)
        iterator = dataset.make_one_shot_iterator()
        prepared_batch = iterator.get_next()

        # Summaries for checking that the loading and data augmentation goes fine
        if image_summaries:
            shape_summary_img = tf.cast(tf.shape(prepared_batch['images'])[1:3]/3, tf.int32)
            tf.summary.image('input/image',
                             tf.image.resize_images(prepared_batch['images'], shape_summary_img),
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
                                 tf.image.resize_images(label_export, shape_summary_img), max_outputs=1)
            if 'weight_maps' in prepared_batch:
                tf.summary.image('input/weight_map',
                                 tf.image.resize_images(prepared_batch['weight_maps'][:, :, :, None], shape_summary_img),
                                 max_outputs=1)

        return prepared_batch, prepared_batch.get('labels')

    return fn


def serving_input_filename(resized_size):
    def serving_input_fn():
        # define placeholder for filename
        filename = tf.placeholder(dtype=tf.string)

        # TODO : make it batch-compatible (with Dataset or string input producer)
        image = load_and_resize_image(filename, 3, resized_size)

        image_batch = image[None]
        features = {'images': image_batch}

        receiver_inputs = {'filename': filename}

        input_from_resized_images = {'resized_images': image_batch}

        return tf.estimator.export.ServingInputReceiver(features, receiver_inputs,
                                                        receiver_tensors_alternatives={'from_resized_images':
                                                                                           input_from_resized_images})

    return serving_input_fn


def serving_input_image():
    dic_input_serving = {'images': tf.placeholder(tf.float32, [None, None, None, 3])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(dic_input_serving)


def local_entropy(tf_binary_img: tf.Tensor, sigma=3):
    tf_binary_img.get_shape().assert_has_rank(2)
    def get_gaussian_filter_1d(sigma):
        sigma_r = int(np.round(sigma))
        x = np.zeros(6 * sigma_r + 1, dtype=np.float32)
        x[3 * sigma_r] = 1
        return ndimage.filters.gaussian_filter(x, sigma=sigma)

    def _fn(img):
        labelled, nb_components = ndimage.measurements.label(img)
        lut = np.concatenate(
            [np.array([0], np.int32), np.random.randint(20, size=nb_components + 1, dtype=np.int32) + 1])
        output = lut[labelled]
        return output

    label_components = tf.py_func(_fn, [tf_binary_img], tf.int32)
    label_components.set_shape([None, None])
    one_hot_components = tf.one_hot(label_components, tf.reduce_max(label_components))
    one_hot_components = tf.transpose(one_hot_components, [2, 0, 1])

    local_components_avg = tf.nn.conv2d(one_hot_components[:, :, :, None],
                                        get_gaussian_filter_1d(sigma)[None, :, None, None], (1, 1, 1, 1),
                                        padding='SAME')
    local_components_avg = tf.nn.conv2d(local_components_avg, get_gaussian_filter_1d(sigma)[:, None, None, None],
                                        (1, 1, 1, 1), padding='SAME')
    local_components_avg = tf.transpose(local_components_avg[:, :, :, 0], [1, 2, 0])
    local_components_avg = tf.pow(local_components_avg, 1 / 1.4)
    local_components_avg = local_components_avg/(tf.reduce_sum(local_components_avg, axis=2, keep_dims=True)+1e-6)
    return -tf.reduce_sum(local_components_avg * tf.log(local_components_avg + 1e-6), axis=2)
