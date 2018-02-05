#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
import json
from doc_seg_datasets.evaluation import dibco_evaluate_epoch
import argparse
import sys
from hashlib import sha1
from tqdm import tqdm


PARAMS_POST_PROCESSING_LIST = [
    {'threshold': -1},
    {'threshold': 0.4},
    {'threshold': 0.5},
    {'threshold': 0.6},
    {'threshold': 0.7}
]
POST_PROCESSING_DIR_NAME = 'post_processing'


def _hash_dict(params):
    return sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


def evaluate_one_dibco_model(model_dir, labels_dir, post_processing_params, verbose=False, save_params=True):

    eval_outputs_dir = os.path.join(model_dir, 'eval', 'epoch_*')
    list_saved_epochs = glob(eval_outputs_dir)
    if len(list_saved_epochs) == 0:
        print('No file found in : {}'.format(eval_outputs_dir))
        return

    post_process_dir = os.path.join(model_dir, POST_PROCESSING_DIR_NAME, _hash_dict(post_processing_params))
    os.makedirs(post_process_dir, exist_ok=True)

    measures = list()
    for saved_epoch in list_saved_epochs:
        epoch_dir_name = saved_epoch.split(os.path.sep)[-1]
        measures.append((int(epoch_dir_name.split('_')[1]),
                         dibco_evaluate_epoch(saved_epoch, labels_dir,
                                              verbose=verbose, threshold=post_processing_params['threshold'])))

    validation_scores = {epoch: {k: v for k, v in vars(metric).items()
                                 if k in ['MSE', 'PSNR', 'precision', 'recall', 'f_measure']}
                         for (epoch, metric) in measures}

    with open(os.path.join(post_process_dir, 'validation_scores.json'), 'w') as f:
        json.dump(validation_scores, f)

    if save_params:
        with open(os.path.join(post_process_dir, 'post_process_params.json'), 'w') as f:
            json.dump(post_processing_params, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-dir', type=str, required=True, nargs='+')
    parser.add_argument('-l', '--labels-dir', type=str, required=True)
    parser.add_argument('-p', '--params-json-file', type=str, default=None)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = vars(parser.parse_args())

    if args.get('params_json_file') is None:
        params_list = PARAMS_POST_PROCESSING_LIST
    else:
        with open(args.get('params_json_file'), 'r') as f:
            params_list = [json.load(f)]

    model_dirs = args.get('model_dir')
    print('Found {} configs and {} model directories'.format(len(params_list), len(model_dirs)))

    for params in tqdm(params_list):
        for model_dir in tqdm(model_dirs):
            evaluate_one_dibco_model(model_dir, args.get('labels_dir'), params,
                                     args.get('verbose'))

