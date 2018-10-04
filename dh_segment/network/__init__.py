_MODEL = [
    'inference_vgg16',
    'inference_resnet_v1_50',
    'inference_u_net',
    'vgg_16_fn',
    'resnet_v1_50_fn'
]

__all__ = _MODEL

from .model import *
from .pretrained_models import *
