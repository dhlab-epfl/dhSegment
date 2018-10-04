Demo
----

This demo shows the usage of dhSegment for page document extraction.
It trains a model from scratch (optional) using the READ-BAD dataset :cite:`gruning2018read`
and the annotations of `Pagenet`_ :cite:`tensmeyer2017pagenet` (annotator1 is used).
In order to limit memory usage, the images in the dataset we provide have been downsized to have 1M pixels each.

.. _Pagenet: https://github.com/ctensmeyer/pagenet/tree/master/annotations


**How to**

1. Get the annotated dataset `here`_, which already contains the folders ``images`` and ``labels``
for training, validation and testing set. Unzip it into ``model/pages``. ::

    cd demo/
    wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip
    unzip pages.zip
    cd ..

.. _here: https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip

2. (Only needed if training from scratch) Download the pretrained weights for ResNet : ::

    cd pretrained_models/
    python download_resnet_pretrained_model.py
    cd ..

3. You can train the model from scratch with: ``python train.py with demo/demo_config.json``
but because this takes quite some time, we recommend you to skip this and just download the
`provided model`_ (download and unzip it in ``demo/model``) ::

    cd demo/
    wget https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip
    unzip model.zip
    cd ..

.. _provided model : https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip

4. (Only if training from scratch) You can visualize the progresses in tensorboard by running
``tensorboard --logdir .`` in the ``demo`` folder.

5. Run ``python demo.py``

6. Have a look at the results in ``demo/processed_images``

