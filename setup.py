#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment',
      version='0.4.0',
      license='GPL',
      url='https://github.com/dhlab-epfl/dhSegment',
      description='Generic framework for historical document processing',
      packages=find_packages(exclude=['exps*']),
      project_urls={
          'Paper': 'https://arxiv.org/abs/1804.10371',
          'Source Code': 'https://github.com/dhlab-epfl/dhSegment'
      },
      scripts=['dh_segment_train'],
      install_requires=[
        #'tensorflow-gpu==1.13.1',
        'numpy==1.16.2',
        'imageio==2.5.0',
        'pandas==0.24.2',
        'scipy==1.2.1',
        'shapely==1.6.4',
        'scikit-learn==0.20.3',
        'scikit-image==0.15.0',
        'opencv-python==4.0.1.23',
        'tqdm==4.31.1',
        'sacred==0.7.4',
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
