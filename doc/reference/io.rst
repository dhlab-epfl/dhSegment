.. comment
    Interface
    =========

    Input functions for ``tf.Estimator``
    ------------------------------------

    Input function

    .. autosummary::
        input.input_fn

    Data augmentation

    .. autosummary::
        data_augmentation_fn
        extract_patches_fn
        rotate_crop

    Resizing function

    .. autosummary::
        dh_segment.io.input_utils.resize_image
        dh_segment.io.input_utils.load_and_resize_image


    Tensorflow serving functions
    ----------------------------

    .. autosummary::
        dh_segment.io.input.serving_input_filename
        dh_segment.io.input.serving_input_image

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

==============
Input / Output
==============

.. automodule:: dh_segment.io
    :members:
    :undoc-members:

.. automodule:: dh_segment.io.PAGE
    :members:
    :undoc-members:
