#!/usr/bin/env python

import urllib.request
import tarfile
import os

if __name__ == '__main__':
    tar_filename = 'resnet_v1_50.tar.gz'
    urllib.request.urlretrieve('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz', tar_filename)
    print('Start downloading')
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()
    print('Resnet pre-trained weights downloaded!')
    os.remove(tar_filename)
