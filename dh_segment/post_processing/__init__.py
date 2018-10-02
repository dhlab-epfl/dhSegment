_BINARIZATION = [
    'thresholding',
    'cleaning_binary',

]

_EVALUATION = [
    'Metrics',
    'intersection_over_union'
]

_PAGE_OBJECTS = [
    'Point',
    'Text',
    'Region',
    'TextLine',
    'GraphicRegion',
    'TextRegion',
    'TableRegion',
    'SeparatorRegion',
    'Border',
    'Metadata',
    'GroupSegment',
    'Page'
]

_PAGE_FN = [
    'parse_file',
    'json_serialize'
]

_DETECTION = [
    'find_boxes',
    'find_polygonal_regions'
]

_VECTORIZATION = [
    'find_lines'
]

__all__ = _BINARIZATION + _DETECTION + _EVALUATION + _PAGE_FN + _PAGE_OBJECTS + _VECTORIZATION

from .binarization import *
from .boxes_detection import *
from .evaluation import *
from .line_vectorization import *
from .PAGE import *
from .polygon_detection import *
