Installation
------------

Using ``pip``
^^^^^^^^^^^^^

1. Clone the repository using ``git clone https://github.com/dhlab-epfl/dhSegment.git``

2. Create and activate a virtualenv ::

        virtualenv myvirtualenvs/dh_segment
        source myvirtualenvs/dh_segment/bin/activate

3. Install the dependencies using ``pip`` (this will look for the ``setup.py`` file) ::

        pip install git+https://github.com/dhlab-epfl/dhSegment

Using Anaconda
^^^^^^^^^^^^^^

1. Install Anaconda or Miniconda (`installation procedure <https://conda.io/docs/user-guide/install/index.html#>`_)

2. Clone the repository: ``git clone https://github.com/dhlab-epfl/dhSegment.git``

3. Create a virtual environment with all the packages: ``conda env create -f environment.yml``

4. Then activate the environment with ``source activate dh_segment``

5. It might be possible that the following needs to be added to your ``~/.bashrc`` ::

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda

6. To be able to import the package (i.e ``import dh_segment``) in your code, you have to run: ::

    python setup.py install

