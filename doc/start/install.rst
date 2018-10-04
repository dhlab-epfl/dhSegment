Installation
------------

Using Anaconda
^^^^^^^^^^^^^^

- Install Anaconda or Miniconda

- Create a virtual environment with all the packages ``conda env create -f environment.yml``

- Then activate the environment with ``source activate dh_segment``

- It might be possible that the following needs to be added to your ``~/.bashrc`` ::

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
    export CUDA_HOME=/usr/local/cuda

- To be able to import the package (i.e ``import dh_segment``) in your code, you have to run : ::

    python setup.py install


Using ``pip``
^^^^^^^^^^^^^