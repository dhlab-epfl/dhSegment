import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
from scipy import ndimage
import numpy as np


def data_augmentation_fn(input_image: tf.Tensor, label_image: tf.Tensor,
                         flip_lr: bool=True, flip_ud: bool=True, color: bool=True) -> (tf.Tensor, tf.Tensor):
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
        if color:
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


def resize_image(image: tf.Tensor, size: int, interpolation='BILINEAR'):
    with tf.name_scope('ImageRescaling'):
        input_shape = tf.cast(tf.shape(image)[:2], tf.float32)
        size = tf.cast(size, tf.float32)
        # Compute new shape
        # We want X/Y = x/y and we have size = x*y so :
        ratio = tf.div(input_shape[1], input_shape[0])
        new_height = tf.sqrt(tf.div(size, ratio))
        new_width = tf.div(size, new_height)
        new_shape = tf.cast([new_height, new_width], tf.int32)
        resize_method = {
            'NEAREST': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            'BILINEAR': tf.image.ResizeMethod.BILINEAR
        }
        return tf.image.resize_images(image, new_shape, method=resize_method[interpolation])


def load_and_resize_image(filename, channels, size=None, interpolation='BILINEAR') -> tf.Tensor:
    """

    :param filename: string tensor
    :param channels: nb of channels for the decoded image
    :param size: number of desired pixels in the resized image, tf.Tensor or int (None for no resizing)
    :param interpolation:
    :param return_original_shape: returns the original shape of the image before resizing if this flag is True
    :return: decoded and resized float32 tensor [h, w, channels],
    """
    with tf.name_scope('load_img'):
        decoded_image = tf.to_float(tf.image.decode_jpeg(tf.read_file(filename), channels=channels,
                                                         try_recover_truncated=True))
        # TODO : if one side is smaller than size of patches (and make patches == true),
        # TODO : force the image to have at least patch size
        if size is not None and not(isinstance(size, int) and size <= 0):
            result_image = resize_image(decoded_image, size, interpolation)
        else:
            result_image = decoded_image

        return result_image


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
        return tf.reshape(patches, [tf.reduce_prod(patches_shape[:3]), h, w, int(c)])  # returns [batch_patches, h, w, c]


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
    local_components_avg = local_components_avg / (tf.reduce_sum(local_components_avg, axis=2, keep_dims=True) + 1e-6)
    return -tf.reduce_sum(local_components_avg * tf.log(local_components_avg + 1e-6), axis=2)
