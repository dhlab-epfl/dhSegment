#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment',
      version='0.5.0',
      license='GPL',
      url='https://github.com/dhlab-epfl/dhSegment',
      description='Generic framework for historical document processing',
      packages=find_packages(),
      project_urls={
          'Paper': 'https://arxiv.org/abs/1804.10371',
          'Source Code': 'https://github.com/dhlab-epfl/dhSegment'
      },
      install_requires=[
        'imageio>=2.5',
        'pandas>=0.24.2',
        'shapely>=1.6.4',
        'scikit-learn>=0.20.3',
        'scikit-image>=0.15.0',
        'opencv-python>=4.0.1',
        'tqdm>=4.31.1',
        'sacred==0.7.4',  # 0.7.5 causes an error
        'requests>=2.21.0',
        'click>=7.0'
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
