==============
Input / Output
==============

Interface
=========

Input function for ``tf.Estimator``
-----------------------------------
.. autosummary::
    dh_segment.io.input_fn

----

Tensorflow serving functions
----------------------------

.. autosummary::
    dh_segment.io.serving_input_filename
    dh_segment.io.serving_input_image

----

PAGE XML and JSON import / export
---------------------------------

PAGE classes

.. autosummary::
    dh_segment.io.PAGE.Point
    dh_segment.io.PAGE.Text
    dh_segment.io.PAGE.Border
    dh_segment.io.PAGE.TextRegion
    dh_segment.io.PAGE.TextLine
    dh_segment.io.PAGE.GraphicRegion
    dh_segment.io.PAGE.TableRegion
    dh_segment.io.PAGE.SeparatorRegion
    dh_segment.io.PAGE.GroupSegment
    dh_segment.io.PAGE.Metadata
    dh_segment.io.PAGE.Page

Abstract classes

.. autosummary::
    dh_segment.io.PAGE.BaseElement
    dh_segment.io.PAGE.Region

Parsing and helpers

.. autosummary::
    dh_segment.io.PAGE.parse_file
    dh_segment.io.PAGE.json_serialize

----

.. automodule:: dh_segment.io
    :members:
    :undoc-members:

.. automodule:: dh_segment.io.PAGE
    :members:
    :undoc-members:
