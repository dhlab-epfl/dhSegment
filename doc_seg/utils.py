import tensorflow as tf
import numpy as np
import os
import json
import pickle


class PredictionType:
    CLASSIFICATION = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'
    MULTILABEL = 'MULTILABEL'

    @classmethod
    def parse(cls, prediction_type):
        if prediction_type == 'CLASSIFICATION':
            return PredictionType.CLASSIFICATION
        elif prediction_type == 'REGRESSION':
            return PredictionType.REGRESSION
        elif prediction_type == 'MULTILABEL':
            return PredictionType.MULTILABEL
        else:
            raise NotImplementedError('Unknown prediction type : {}'.format(prediction_type))


class BaseParams:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        result = cls()
        keys = result.to_dict().keys()
        for k, v in d.items():
            assert k in keys, k
            setattr(result, k, v)
        result.check_params()
        return result

    def check_params(self):
        pass


class VGG16ModelParams:
    PRETRAINED_MODEL_FILE = '/home/pretrained_models/vgg_16.ckpt'
    INTERMEDIATE_CONV = [
        [(256, 3)]
    ]
    UPSCALE_PARAMS = [
        [(32, 3)],
        [(64, 3)],
        [(128, 3)],
        [(256, 3)],
        [(512, 3)],
        [(512, 3)]
    ]
    SELECTED_LAYERS_UPSCALING = [
        True,
        True,  # Must have same length as vgg_upscale_params
        True,
        True,
        False,
        False
    ]
    CORRECTED_VERSION = None


class ResNetModelParams:
    PRETRAINED_MODEL_FILE = '/home/pretrained_models/resnet_v1_50.ckpt'
    INTERMEDIATE_CONV = None
    UPSCALE_PARAMS = [
        # (Filter size (depth bottleneck's output), number of bottleneck)
        (32, 2),
        (64, 1),
        (256, 1),
        (512, 1),
        (512, 1)
    ]
    SELECTED_LAYERS_UPSCALING = [
        # Must have the same length as resnet_upscale_params
        True,
        True,
        True,
        True,
        True
    ]
    CORRECT_VERSION = False


class UNetModelParams:
    PRETRAINED_MODEL_FILE = None
    INTERMEDIATE_CONV = None
    UPSCALE_PARAMS = None
    SELECTED_LAYERS_UPSCALING = None
    CORRECT_VERSION = False


class ModelParams(BaseParams):
    def __init__(self, **kwargs):
        self.batch_norm = kwargs.get('batch_norm', True)  # type: bool
        self.batch_renorm = kwargs.get('batch_renorm', True)  # type: bool
        self.weight_decay = kwargs.get('weight_decay', 1e-5)  # type: float
        self.n_classes = kwargs.get('n_classes', None)  # type: int
        self.pretrained_model_name = kwargs.get('pretrained_model_name', None)  # type: str
        self.max_depth = kwargs.get('max_depth', 512)  # type: int

        if self.pretrained_model_name == 'vgg16':
            model_class = VGG16ModelParams
        elif self.pretrained_model_name == 'resnet50':
            model_class = ResNetModelParams
        elif self.pretrained_model_name == 'unet':
            model_class = UNetModelParams
        else:
            raise NotImplementedError

        self.pretrained_model_file = kwargs.get('pretrained_model_file', model_class.PRETRAINED_MODEL_FILE)
        self.intermediate_conv = kwargs.get('intermediate_conv', model_class.INTERMEDIATE_CONV)
        self.upscale_params = kwargs.get('upscale_params', model_class.UPSCALE_PARAMS)
        self.selected_levels_upscaling = kwargs.get('selected_levels_upscaling', model_class.SELECTED_LAYERS_UPSCALING)
        self.correct_resnet_version = kwargs.get('correct_resnet_version', model_class.CORRECT_VERSION)
        self.check_params()

    def check_params(self):
        # Pretrained model name check
        # assert self.upscale_params is not None and self.selected_levels_upscaling is not None, \
        #     'Model parameters cannot be None'
        if self.upscale_params is not None and self.selected_levels_upscaling is not None:

            assert len(self.upscale_params) == len(self.selected_levels_upscaling), \
                'Upscaling levels and selection levels must have the same lengths (in model_params definition), ' \
                '{} != {}'.format(len(self.upscale_params),
                                  len(self.selected_levels_upscaling))

            assert os.path.isfile(self.pretrained_model_file), \
                'Pretrained weights file {} not found'.format(self.pretrained_model_file)


class TrainingParams(BaseParams):
    def __init__(self):
        self.n_epochs = 20
        self.evaluate_every_epoch = 10
        self.learning_rate = 1e-5
        self.exponential_learning = True
        self.batch_size = 5
        self.data_augmentation = False
        self.data_augmentation_flip_lr = False
        self.data_augmentation_flip_ud = False
        self.data_augmentation_color = False
        self.data_augmentation_max_rotation = 0.2  # in radians
        self.data_augmentation_max_scaling = 0.05  # range : [0, 1]
        self.make_patches = True
        self.patch_shape = (300, 300)
        # self.input_resized_shape = (480, 320)  # Deprecated
        # If input_resized_size == -1, no resizing is done
        self.input_resized_size = int(72e4)  # (600*1200) # type: int
        self.weights_labels = None
        self.training_margin = 16
        self.local_entropy_ratio = 0.0
        self.local_entropy_sigma = 3
        self.focal_loss_gamma = 0.0

    def check_params(self):
        assert self.training_margin*2 < min(self.patch_shape)


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


def get_image_shape_tensor(tensor: tf.Tensor):
    if tensor.get_shape()[1].value and \
                    tensor.get_shape()[2].value:
        target_shape = tensor.get_shape()[1:3]
    else:
        target_shape = tf.shape(tensor)[1:3]
    return target_shape


def parse_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename, dict):
    with open(filename, 'w') as f:
        json.dump(dict, f, indent=4, sort_keys=True)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_pickle(filename, obj):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)
