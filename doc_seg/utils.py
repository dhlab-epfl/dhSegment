import tensorflow as tf
import numpy as np
import os
import json


class PredictionType:
    CLASSIFICATION = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'
    MULTILABEL = 'MULTILABEL'


class Params:
    def __init__(self, **kwargs):
        self._n_epochs = kwargs.get('n_epochs', 20)
        self._evaluate_every_epoch = kwargs.get('evaluate_every_epochs', 5)
        self._learning_rate = kwargs.get('learning_rate', 1e-5)
        self._exponential_learning = kwargs.get('exponential_learning', True)
        self._batch_size = kwargs.get('batch_size', 5)
        self._batch_norm = kwargs.get('batch_norm', False)
        self._batch_renorm = kwargs.get('batch_renorm', False)
        self._weight_decay = kwargs.get('weight_decay', 1e-5)
        self._data_augmentation = kwargs.get('data_augmentation', True)
        self._make_patches = kwargs.get('make_patches', False)
        self._patch_shape = kwargs.get('patch_shape', (300, 300))
        self._input_resized_size = kwargs.get('resized_size', (480, 320))
        self._input_dir_train = kwargs.get('input_dir_train')
        self._input_dir_eval = kwargs.get('input_dir_eval')
        self._output_model_dir = kwargs.get('output_model_dir')
        self._gpu = kwargs.get('gpu', '')
        self._class_file = kwargs.get('class_file')
        self._pretrained_model_name = kwargs.get('model_name')
        # self._pretrained_model_file = kwargs.get('pretrained_file')
        self._prediction_type = self._set_prediction_type(kwargs.get('prediction_type'))
        self._vgg_intermediate_conv = kwargs.get('vgg_intermediate_conv')
        self._vgg_upscale_params = kwargs.get('vgg_upscale_params')
        self._vgg_selected_levels_upscaling = kwargs.get('vgg_selected_levels_upscaling')

        # Prediction type check
        if self._prediction_type == PredictionType.CLASSIFICATION:
            assert self._class_file is not None, 'Unable to find a valid classes.txt file'
        elif self._prediction_type == PredictionType.REGRESSION:
            self._n_classes = 1
        elif self._prediction_type == PredictionType.MULTILABEL:
            assert self._class_file is not None, 'Unable to find a valid classes.txt file'

        # Pretrained model name check
        if self._pretrained_model_name == 'vgg16':
            assert self._vgg_upscale_params is not None and self._vgg_selected_levels_upscaling is not None, \
                'VGG16 parameters cannot be None'

            assert len(self._vgg_upscale_params) == len(self._vgg_selected_levels_upscaling), \
                'Upscaling levels and selection levels must have the same lengths (in model_params definition), ' \
                '{} != {}'.format(len(self._vgg_upscale_params),
                                  len(self._vgg_selected_levels_upscaling))

            self._pretrained_model_file = '/mnt/cluster-nas/benoit/pretrained_nets/vgg_16.ckpt'
        else:
            raise NotImplementedError('Unknown model {}'.format(self._pretrained_model_name))

    def _set_prediction_type(self, prediction_type):
            if prediction_type == 'CLASSIFICATION':
                return PredictionType.CLASSIFICATION
            elif prediction_type == 'REGRESSION':
                return PredictionType.REGRESSION
            elif prediction_type == 'MULTILABEL':
                return PredictionType.MULTILABEL
            else:
                raise NotImplementedError('Unknown prediction type : {}'.format(prediction_type))

    def export_experiment_params(self):
        if not os.path.isdir(self.output_model_dir):
            os.mkdir(self.output_model_dir)
        with open(os.path.join(self.output_model_dir, 'model_params.json'), 'w') as f:
            json.dump(vars(self), f)

    @property
    def output_model_dir(self):
        return self._output_model_dir

    @property
    def input_dir_train(self):
        return self._input_dir_train

    @property
    def input_dir_eval(self):
        return self._input_dir_eval

    @property
    def class_file(self):
        return self._class_file

    @property
    def prediction_type(self):
        return self._prediction_type

    @property
    def gpu(self):
        return self._gpu

    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value):
        self._n_classes = value

    @property
    def input_resized_size(self):
        return self._input_resized_size

    @property
    def evaluate_every_epoch(self):
        return self._evaluate_every_epoch

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_norm(self):
        return self._batch_norm

    @property
    def batch_renorm(self):
        return self._batch_renorm

    @property
    def data_augmentation(self):
        return self._data_augmentation

    @property
    def make_patches(self):
        return self._make_patches

    @property
    def patch_shape(self):
        return self._patch_shape

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def exponential_learning(self):
        return self._exponential_learning

    @property
    def vgg_intermediate_conv(self):
        return self._vgg_intermediate_conv

    @property
    def vgg_upscale_params(self):
        return self._vgg_upscale_params

    @property
    def vgg_selected_levels_upscaling(self):
        return self._vgg_selected_levels_upscaling

    @property
    def pretrained_model_file(self):
        return self._pretrained_model_file

    @property
    def pretrained_model_name(self):
        return self._pretrained_model_name


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
    classes_color_values = get_classes_color_from_file(classes_file)
    # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
    with tf.name_scope('MultiLabelAssign'):
        if len(label_image.get_shape()) == 3:
            diff = tf.cast(label_image[:, :, None, :], tf.float32) - tf.constant(classes_color_values[None, None, :, :])  # [H,W,C,3]
        elif len(label_image.get_shape()) == 4:
            diff = tf.cast(label_image[:, :, :, None, :], tf.float32) - tf.constant(
                classes_color_values[None, None, None, :, :])  # [B,H,W,C,3]
        else:
            raise NotImplementedError('Length is : {}'.format(len(label_image.get_shape())))

        pixel_class_min = tf.reduce_min(diff, axis=-1)  # [H,W,C] or [B,H,W,C]
        multi_class_label = tf.greater_equal(pixel_class_min, 0)  # [H,W,C] or [B,H,W,C] with TRUE, FALSE
        one_hot_multiclass = tf.stack([multi_class_label,
                                       tf.logical_not(multi_class_label)], axis=-1)  # [H,W,C,2] 'one_hot' encoding
        shape = one_hot_multiclass.get_shape().as_list()
        if len(one_hot_multiclass.get_shape()) == 4:
            new_shape = (shape[0], shape[1], shape[2] * shape[3])
            reshaped_label = tf.reshape(one_hot_multiclass, new_shape)  # [H,W,C*2]
            reshaped_label.set_shape(new_shape)
        elif len(one_hot_multiclass.get_shape()) == 5:
            new_shape = (shape[0], shape[1], shape[2], shape[3] * shape[4])
            reshaped_label = tf.reshape(one_hot_multiclass, new_shape)  # [B,H,W,C*2]
            reshaped_label.set_shape(new_shape)
        else:
            raise NotImplementedError('Length is : {}'.format(len(one_hot_multiclass.get_shape())))

        return tf.cast(reshaped_label, dtype=tf.float32)


def multiclass_to_label_image(class_label: tf.Tensor, classes_file: str) -> tf.Tensor:
    classes_color_values = get_classes_color_from_file(classes_file)
    with tf.name_scope('Label2Image'):
        shape = class_label.get_shape()

        # class_label : [B,H,W,C*2]
        new_shape = tf.concat([shape[:-1], [classes_color_values.shape[0], 2]], axis=0)
        labels_reshaped = tf.reshape(class_label, new_shape)  # [B,H,W,C,2]

        if classes_color_values.shape[0] == 2:
            if len(labels_reshaped.get_shape()) == 4:
                labels_formatted = labels_reshaped[:, :, :, 1]
                label_img = tf.concat([labels_formatted, tf.zeros(shape[:-1], dtype=tf.float32)[:, :, None]], axis=-1)
            elif len(labels_reshaped.get_shape()) == 5:
                labels_formatted = labels_reshaped[:, :, :, :, 1]
                label_img = tf.concat([labels_formatted, tf.zeros(shape[:-1], dtype=tf.float32)[:, :, :, None]], axis=-1)
            else:
                raise NotImplementedError('Length is : {}'.format(len(labels_reshaped.get_shape())))
            return tf.cast(label_img, dtype=tf.int32)
        else:
            # TODO better
            return tf.cast(class_label[:, :, :, :3, 0], dtype=tf.int32)


def get_classes_color_from_file(classes_file: str) -> np.ndarray:
    if not os.path.exists(classes_file):
        raise FileNotFoundError(classes_file)
    result = np.loadtxt(classes_file).astype(np.float32)
    assert result.shape[1] == 3, "Color file should represent RGB values"
    return result