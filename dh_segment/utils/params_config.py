#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from .misc import get_class_from_name
from ..network.model import Encoder, Decoder
from typing import Type


class PredictionType:
    """

    :cvar CLASSIFICATION:
    :cvar REGRESSION:
    :cvar MULTILABEL:
    """
    CLASSIFICATION = 'CLASSIFICATION'
    REGRESSION = 'REGRESSION'
    MULTILABEL = 'MULTILABEL'

    @classmethod
    def parse(cls, prediction_type) -> 'PredictionType':
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
    """
    Parameters related to the model
    :param encoder_name:
    :param encoder_params:
    :param decoder_name:
    :param decoder_params:
    :param n_classes:
    """
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
    """Parameters to configure training process

    :ivar n_epochs: number of epoch for training
    :vartype n_epochs: int
    :ivar evaluate_every_epoch: the model will be evaluated every `n` epochs
    :vartype evaluate_every_epoch: int
    :ivar learning_rate: the starting learning rate value
    :vartype learning_rate: float
    :ivar exponential_learning: option to use exponential learning rate
    :vartype exponential_learning: bool
    :ivar batch_size: size of batch
    :vartype batch_size: int
    :ivar data_augmentation: option to use data augmentation (by default is set to False)
    :vartype data_augmentation: bool
    :ivar data_augmentation_flip_lr: option to use image flipping in right-left direction
    :vartype data_augmentation_flip_lr: bool
    :ivar data_augmentation_flip_ud: option to use image flipping in up down direction
    :vartype data_augmentation_flip_ud: bool
    :ivar data_augmentation_color: option to use data augmentation with color
    :vartype data_augmentation_color: bool
    :ivar data_augmentation_max_rotation: maximum angle of rotation (in radians) for data augmentation
    :vartype data_augmentation_max_rotation: float
    :ivar data_augmentation_max_scaling: maximum scale of zooming during data augmentation (range: [0,1])
    :vartype data_augmentation_max_scaling: float
    :ivar make_patches: option to crop image into patches. This will cut the entire image in several patches
    :vartype make_patches: bool
    :ivar patch_shape: shape of the patches
    :vartype patch_shape: tuple
    :ivar input_resized_size: size (in pixel) of the image after resizing. The original ratio is kept. If no resizing \
    is wanted, set it to -1
    :vartype input_resized_size: int
    :ivar weights_labels: weight given to each label. Should be a list of length = number of classes
    :vartype weights_labels: list
    :ivar training_margin: size of the margin to add to the images. This is particularly useful when training with \
    patches
    :vartype training_margin: int
    :ivar local_entropy_ratio:
    :vartype local_entropy_ratio: float
    :ivar local_entropy_sigma:
    :vartype local_entropy_sigma: float
    :ivar focal_loss_gamma: value of gamma for the focal loss. See paper : https://arxiv.org/abs/1708.02002
    :vartype focal_loss_gamma: float
    """
    def __init__(self, **kwargs):
        self.n_epochs = kwargs.get('n_epochs', 20)
        self.evaluate_every_epoch = kwargs.get('evaluate_every_epoch', 10)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.exponential_learning = kwargs.get('exponential_learning', True)
        self.batch_size = kwargs.get('batch_size', 5)
        self.data_augmentation = kwargs.get('data_augmentation', False)
        self.data_augmentation_flip_lr = kwargs.get('data_augmentation_flip_lr', False)
        self.data_augmentation_flip_ud = kwargs.get('data_augmentation_flip_ud', False)
        self.data_augmentation_color = kwargs.get('data_augmentation_color', False)
        self.data_augmentation_max_rotation = kwargs.get('data_augmentation_max_rotation', 0.2)
        self.data_augmentation_max_scaling = kwargs.get('data_augmentation_max_scaling', 0.05)
        self.make_patches = kwargs.get('make_patches', True)
        self.patch_shape = kwargs.get('patch_shape', (300, 300))
        self.input_resized_size = int(kwargs.get('input_resized_size', 72e4))  # (600*1200)
        self.weights_labels = kwargs.get('weights_labels')
        self.weights_evaluation_miou = kwargs.get('weights_evaluation_miou', None)
        self.training_margin = kwargs.get('training_margin', 16)
        self.local_entropy_ratio = kwargs.get('local_entropy_ratio', 0.)
        self.local_entropy_sigma = kwargs.get('local_entropy_sigma', 3)
        self.focal_loss_gamma = kwargs.get('focal_loss_gamma', 0.)

    def check_params(self) -> None:
        """Checks if there is no parameter inconsistency
        """
        assert self.training_margin*2 < min(self.patch_shape)
