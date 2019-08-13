Installation
------------

It is recommended to install ``tensorflow`` (or ``tensorflow-gpu``) independently using Anaconda distribution,
in order to make sure all dependencies are properly installed.

1. Clone the repository using ``git clone https://github.com/dhlab-epfl/dhSegment.git``

2. Install Anaconda or Miniconda (`installation procedure <https://conda.io/docs/user-guide/install/index.html#>`_)

3. Create a virtual environment and activate it ::

        conda create -n dh_segment python=3.6
        source activate dh_segment


4. Install dhSegment dependencies with ``pip install git+https://github.com/dhlab-epfl/dhSegment``

5. Install TensorFlow 1.13 with conda ``conda install tensorflow-gpu=1.13.1``.
