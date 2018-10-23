r"""
The :mod:`dh_segment.inference` module implements the function related to the usage of a dhSegment model,
for instance to use a trained model to inference on new data.

Loading a model
---------------

.. autosummary::
    LoadedModel
"""

__all__ = ['LoadedModel']

from .loader import *