r"""
The :mod:`dh_segment.utils` module contains the parameters for config with `sacred`_ package,
image label vizualization functions and miscelleanous helpers.

Parameters
----------

.. autosummary::
    ModelParams
    TrainingParams

Label image helpers
-------------------

.. autosummary::
    label_image_to_class
    class_to_label_image
    multilabel_image_to_class
    multiclass_to_label_image
    get_classes_color_from_file
    get_n_classes_from_file
    get_classes_color_from_file_multilabel
    get_n_classes_from_file_multilabel

Evaluation utils
----------------

.. autosummary::
    Metrics
    intersection_over_union

Miscellaneous helpers
---------------------

.. autosummary::
    parse_json
    dump_json
    load_pickle
    dump_pickle
    hash_dict

.. _sacred : https://sacred.readthedocs.io/en/latest/index.html

------
"""

_PARAMSCONFIG = [
    'PredictionType',
    'VGG16ModelParams',
    'ResNetModelParams',
    'UNetModelParams',
    'ModelParams',
    'TrainingParams'
]


_LABELS = [
    'label_image_to_class',
    'class_to_label_image',
    'multilabel_image_to_class',
    'multiclass_to_label_image',
    'get_classes_color_from_file',
    'get_n_classes_from_file',
    'get_classes_color_from_file_multilabel',
    'get_n_classes_from_file_multilabel'
]

_MISC = [
    'parse_json',
    'dump_json',
    'load_pickle',
    'dump_pickle',
    'hash_dict'
]

_EVALUATION = [
    'Metrics',
    'intersection_over_union'
]

__all__ = _PARAMSCONFIG + _LABELS + _MISC + _EVALUATION

from .params_config import *
from .labels import *
from .misc import *
from .evaluation import *