=========
Changelog
=========

.. Unreleased
    ----------


0.5.0 - 2019-08-14
------------------

Added
^^^^^

* ``https`` can now be used for PAGEXML schema.
* All the ``PAGE`` objects can now have ``custom_attribute`` (this is mainly for tagging purposes).


Changed
^^^^^^^

* The ``exps`` folders contains now only two examples that can be used as demos. The other experiments have been removed.
* Installation of ``dh_segment`` package is now done via pip (using ``setup.py``) except for ``tensorflow`` package which is installed with anaconda.
* ``setup.py`` has more flexible package versions.
* Forced integer conversion when exporting coordinates to XML format.

Fixed
^^^^^

* In page ``demo.py`` a empty ``Border`` is now created if no region has been detected.


Removed
^^^^^^^

* Experiments in ``exps`` folder have been removed, except for ``page`` and ``cbad``.


0.4.0 - 2019-04-10
------------------

Added
^^^^^

* Input data can be a .csv file with format ``<filename-image>,<filename-label>``.
* ``dh_segment.io.via`` helper functions to generate/export groundtruth from/to VGG Image Annotation tool.
* ``Point.array_to_point`` to export a ``np.array`` into a list of ``Point``.
* PAGEXML Regions can now contain a custom attribute (Transkribus output of region annotation)
* ``Page.to_json()`` method for json formatting.

Changed
^^^^^^^

* ``tensorflow`` v1.13 and ``opencv`` v4.0 are now used.
* mIOU metric for evaluation during training (instead of accuracy).
* TextLines are sorted according to their mean `y` coordinate when exported.

Fixed
^^^^^

* Variable names typos in ``input.py`` and ``train.py``.
* Documentation of the quickstart demo.

Removed
^^^^^^^
