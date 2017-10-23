import tensorflow as tf
import numpy as np
import os


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


class ModelParams(BaseParams):
    def __init__(self):
        self.batch_norm = True  # type: bool
        self.batch_renorm = True  # type: bool
        self.weight_decay = 1e-5  # type: float
        self.n_classes = None  # type: int
        self.pretrained_model_name = 'vgg16'
        self.pretrained_model_file = '/mnt/cluster-nas/benoit/pretrained_nets/vgg_16.ckpt'
        self.vgg_intermediate_conv = [
            [(256, 3)]
        ]
        self.vgg_upscale_params = [
            [(64, 3)],
            [(128, 3)],
            [(256, 3)],
            [(512, 3)],
            [(512, 3)]
        ]
        self.vgg_selected_levels_upscaling = [
            True,  # Must have same length as vgg_upscale_params
            True,
            True,
            False,
            False
        ]

    def check_params(self):
        # Pretrained model name check
        if self.pretrained_model_name == 'vgg16':
            assert self.vgg_upscale_params is not None and self.vgg_selected_levels_upscaling is not None, \
                'VGG16 parameters cannot be None'

            assert len(self.vgg_upscale_params) == len(self.vgg_selected_levels_upscaling), \
                'Upscaling levels and selection levels must have the same lengths (in model_params definition), ' \
                '{} != {}'.format(len(self.vgg_upscale_params),
                                  len(self.vgg_selected_levels_upscaling))
        else:
            raise NotImplementedError('Unknown model {}'.format(self.pretrained_model_name))


class TrainingParams(BaseParams):
    def __init__(self):
        self.n_epochs = 20
        self.evaluate_every_epoch = 5
        self.learning_rate = 1e-5
        self.exponential_learning = True
        self.batch_size = 5
        self.data_augmentation = True
        self.make_patches = False
        self.patch_shape = (300, 300)
        self.input_resized_size = (480, 320)


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
