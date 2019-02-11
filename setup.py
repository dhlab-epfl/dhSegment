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
        'tensorflow',
        'numpy',
        'imageio',
        'pandas',
        'scipy',
        'shapely',
        'scikit-learn',
        'opencv-python',
        'tqdm',
        'requests==2.21.0',
      ],
      extras_require={
          'doc': [
              'sphinx',
              'sphinx-autodoc-typehints',
              'sphinx-rtd-theme',
              'sphinxcontrib-bibtex',
              'sphinxcontrib-websupport'
          ],
      },
      zip_safe=False)
