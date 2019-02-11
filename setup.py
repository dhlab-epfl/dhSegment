#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment',
      version='0.3',
      license='GPL',
      url='https://github.com/dhlab-epfl/dhSegment',
      description='Generic framework for historical document processing',
      packages=find_packages(exclude=['exps*']),
      project_urls={
          'Paper': 'https://arxiv.org/abs/1804.10371',
          'Source Code': 'https://github.com/dhlab-epfl/dhSegment'
      },
      install_requires=[
        'tensorflow-gpu==1.11',
        'numpy==1.14.5',
        'imageio==2.3.0',
        'pandas==0.23.0',
        'scipy==1.1.0',
        'shapely==1.6.4',
        'scikit-learn==0.19.1',
        'scikit-image==0.13.1',
        'opencv-python==3.4.1.15',
        'tqdm==4.23.3',
        'sacred==0.7.3',
        'requests==2.21.0'
      ],
      extras_require={
          'doc': [
              'sphinx==1.8.1',
              'sphinx-autodoc-typehints==1.3.0',
              'sphinx-rtd-theme==0.4.1',
              'sphinxcontrib-bibtex==0.4.0',
              'sphinxcontrib-websupport'
          ],
      },
      zip_safe=False)
