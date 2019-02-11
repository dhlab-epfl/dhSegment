Creating groundtruth data
-------------------------

Using GIMP or Photoshop
^^^^^^^^^^^^^^^^^^^^^^^
Create directly your masks using your favorite image editor. You just have to draw the regions you want to extract
with a different color for each label.

Using VGG Image Annotator (VIA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`VGG Image Annotator (VIA) <http://www.robots.ox.ac.uk/~vgg/software/via/>`_ is an image annotation tool that can be
used to define regions in an image and create textual descriptions of those regions. You can either use it
`online <http://www.robots.ox.ac.uk/~vgg/software/via/via.html>`_ or
`download the application <http://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.5.zip>`_.

From the exported annotations (in JSON format), you'll have to generate the corresponding image masks.
See the :ref:`ref_via` in the ``via`` module.

When assigning attributes to your annotated regions, you should favour attributes of type "dropdown", "checkbox"
and "radio" and avoid "text" type in order to ease the parsing of the exported file (avoid typos and formatting errors).



