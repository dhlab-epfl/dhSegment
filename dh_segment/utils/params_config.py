#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
import warnings
from random import shuffle


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
    PRETRAINED_MODEL_FILE = 'pretrained_models/vgg_16.ckpt'
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
    PRETRAINED_MODEL_FILE = 'pretrained_models/resnet_v1_50.ckpt'
    INTERMEDIATE_CONV = None
    UPSCALE_PARAMS = [
        # (Filter size (depth bottleneck's output), number of bottleneck)
        (32, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (512, 0)
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
        self.weight_decay = kwargs.get('weight_decay', 1e-6)  # type: float
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

            # assert os.path.isfile(self.pretrained_model_file), \
            #     'Pretrained weights file {} not found'.format(self.pretrained_model_file)
            if not os.path.isfile(self.pretrained_model_file):
                warnings.warn('WARNING - Default pretrained weights file in {} was not found. '
                              'Have you changed the default pretrained file ?'.format(self.pretrained_model_file))


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
        # If input_resized_size == -1, no resizing is done
        self.input_resized_size = int(72e4)  # (600*1200) # type: int
        self.weights_labels = None
        self.training_margin = 16
        self.local_entropy_ratio = 0.0
        self.local_entropy_sigma = 3
        self.focal_loss_gamma = 0.0
        self.focal_loss_alpha = -1.0

    def check_params(self):
        assert self.training_margin*2 < min(self.patch_shape)