r"""
The :mod:`dh_segment.post_processing` module contains functions to post-process probability maps.

**Binarization**

.. autosummary::
    thresholding
    cleaning_binary

**Detection**

.. autosummary::
    find_boxes
    find_polygonal_regions

**Vectorization**

.. autosummary::
    find_lines

------

"""

_BINARIZATION = [
    'thresholding',
    'cleaning_binary',

]

_DETECTION = [
    'find_boxes',
    'find_polygonal_regions'
]

_VECTORIZATION = [
    'find_lines'
]

__all__ = _BINARIZATION + _DETECTION + _VECTORIZATION

from .binarization import *
from .boxes_detection import *
from .line_vectorization import *
from .polygon_detection import *

