Training
--------

.. note:: A good nvidia GPU (6GB RAM at least) is most likely necessary to train your own models. We assume CUDA
    and cuDNN are installed.

**Input data**

You need to have your training data in a folder containing ``images`` folder and ``labels`` folder.
The pairs (images, labels) need to have the same name (it is not mandatory to have the same extension file,
however we recommend having the label images as ``.png`` files).

The annotated images in ``label`` folder are (usually) RGB images with the regions to segment annotated with
a specific color.

.. note:: It is now also possible to use a `csv` file  containing the pairs ``original_image_filename``,
    ``label_image_filename`` as input data.

To input a ``csv`` file instead of the two folders ``images`` and ``labels``,
the content should be formatted in the following way: ::

    mypath/myfolder/original_image_filename1,mypath/myfolder/label_image_filename1
    mypath/myfolder/original_image_filename2,mypath/myfolder/label_image_filename2



**The class.txt file**

The file containing the classes has the format shown below, where each row corresponds to one class
(including 'negative' or 'background' class) and each row has 3 values for the 3 RGB values.
Of course each class needs to have a different code. ::

    class.txt

    0 0 0
    0 255 0
    ...


**Config file with ``sacred``**

`sacred`_ package is used to deal with experiments and trainings. Have a look at the documentation to use it properly.

In order to train a model, you should run ``python train.py with <config.json>``

.. _sacred: https://sacred.readthedocs.io/en/latest/quickstart.html