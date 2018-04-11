#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='dh_segment',
      version='0.1',
      url='https://github.com/dhlab-epfl/dhSegment',
      description='Generic framework for historical document processing',
      packages=find_packages(exclude=['exps*']),
      zip_safe=False)
