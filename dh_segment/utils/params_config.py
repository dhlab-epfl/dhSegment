#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from .misc import get_class_from_name
from ..network.model import Encoder, Decoder
from typing import Type


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
        self.encoder_name = kwargs.get('encoder_name', 'dh_segment.network.pretrained_models.ResnetV1_50')  # type: str
        self.encoder_params = kwargs.get('encoder_params', dict())  # type: dict
        self.decoder_name = kwargs.get('decoder_name', 'dh_segment.network.SimpleDecoder')  # type: str
        self.decoder_params = kwargs.get('decoder_params', {
            'upsampling_dims': [32, 64, 128, 256, 512]
        })  # type: dict
        self.n_classes = kwargs.get('n_classes', None)  # type: int

        self.check_params()

    def get_encoder(self) -> Type[Encoder]:
        encoder = get_class_from_name(self.encoder_name)
        assert issubclass(encoder, Encoder), "{} is not an Encoder".format(encoder)
        return encoder

    def get_decoder(self) -> Type[Decoder]:
        decoder = get_class_from_name(self.decoder_name)
        assert issubclass(decoder, Decoder), "{} is not a Decoder".format(decoder)
        return decoder

    def check_params(self):
        self.get_encoder()
        self.get_decoder()


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

    def check_params(self):
        assert self.training_margin*2 < min(self.patch_shape)