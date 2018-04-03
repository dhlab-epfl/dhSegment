#!/usr/bin/env python

import urllib.request
import tarfile
import os
from tqdm import tqdm


def progress_hook(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


if __name__ == '__main__':
    tar_filename = 'vgg_16.tar.gz'
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc="Downloading pre-trained weights") as t:
        urllib.request.urlretrieve('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz', tar_filename,
                                   reporthook=progress_hook(t))
    tar = tarfile.open(tar_filename)
    tar.extractall()
    tar.close()
    print('VGG-16 pre-trained weights downloaded!')
    os.remove(tar_filename)
