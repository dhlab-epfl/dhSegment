#!/usr/bin/env python
__license__ = "GPL"

import tensorflow as tf
import json
import pickle
from hashlib import sha1


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
