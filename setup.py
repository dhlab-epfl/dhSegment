#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment',
      version='0.1',
      license='GPL',
      url='https://github.com/dhlab-epfl/dhSegment',
      description='Generic framework for historical document processing',
      packages=find_packages(exclude=['exps*']),
      project_urls={
          'Paper': 'https://arxiv.org/abs/1804.10371',
          'Source Code': 'https://github.com/dhlab-epfl/dhSegment'
      },
      scripts=['dh_segment_train'],
      zip_safe=False)
