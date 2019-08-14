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

**Example of how to create individual masks from VIA annotation file**

.. code:: python

    from dh_segment.io import via

    collection = 'mycollection'
    annotation_file = 'via_sample.json'
    masks_dir = '/home/project/generated_masks'
    images_dir = './my_images'

    # Load all the data in the annotation file
    # (the file may be an exported project or an export of the annotations)
    via_data = via.load_annotation_data(annotation_file)

    # In the case of an exported project file, you can set ``only_img_annotations=True``
    # to get only the image annotations
    via_annotations = via.load_annotation_data(annotation_file, only_img_annotations=True)

    # Collect the annotated regions
    working_items = via.collect_working_items(via_annotations, collection, images_dir)

    # Collect the attributes and options
    if '_via_attributes' in via_data.keys():
        list_attributes = via.parse_via_attributes(via_data['_via_attributes'])
    else:
        list_attributes = via.get_via_attributes(via_annotations)

    # Create one mask per option per attribute
    via.create_masks(masks_dir, working_items, list_attributes, collection)

