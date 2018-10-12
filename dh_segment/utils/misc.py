#!/usr/bin/env python
__license__ = "GPL"

import tensorflow as tf
import json
import pickle
from hashlib import sha1
from typing import Any
import importlib
import os
import urllib.request
import tarfile
import os
from tqdm import tqdm
from random import shuffle


def parse_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def dump_json(filename, dict):
    with open(filename, 'w') as f:
        json.dump(dict, f, indent=4, sort_keys=True)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_pickle(filename, obj):
    with open(filename, 'wb') as f:
        return pickle.dump(obj, f)


def hash_dict(params):
    return sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


def shuffled(l: list) -> list:
    ll = l.copy()
    shuffle(ll)
    return ll


def get_class_from_name(full_class_name: str) -> Any:
    """
    Tries to load the class from its naming, will import the corresponding module.
    Raises an Error if it does not work.
    :param full_class_name: full name of the class, for instance `foo.bar.Baz`
    :return: the loaded class
    """
    module_name, class_name = full_class_name.rsplit('.', maxsplit=1)
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_data_folder() -> str:
    folder = os.path.join(os.path.expanduser('~'), '.dh_segment')
    os.makedirs(folder, exist_ok=True)
    return folder


def download_file(url: str, output_file: str):
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

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
              desc="Downloading pre-trained weights") as t:
        urllib.request.urlretrieve(url, output_file, reporthook=progress_hook(t))
