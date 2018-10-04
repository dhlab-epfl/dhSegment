_INPUT = [
    'input_fn',
    'serving_input_filename',
    'serving_input_image',
    'data_augmentation_fn',
    'rotate_crop',
    'resize_image',
    'load_and_resize_image',
    'extract_patches_fn',
    'local_entropy'
]

# _PAGE_OBJECTS = [
#     'Point',
#     'Text',
#     'Region',
#     'TextLine',
#     'GraphicRegion',
#     'TextRegion',
#     'TableRegion',
#     'SeparatorRegion',
#     'Border',
#     'Metadata',
#     'GroupSegment',
#     'Page'
# ]
#
# _PAGE_FN = [
#     'parse_file',
#     'json_serialize'
# ]

__all__ = _INPUT  # + _PAGE_OBJECTS + _PAGE_FN

from .input import *
from .input_utils import *
from . import PAGE
