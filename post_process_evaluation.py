#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import json
import os
from glob import glob
from hashlib import sha1
from doc_seg.evaluation import cbad_evaluate_epoch, dibco_evaluate_epoch, cini_evaluate_epoch
from tqdm import tqdm
import better_exceptions

POST_PROCESSING_DIR_NAME = 'post_processing'

POST_PROCESSING_EVAL_FN_DICT = {
    'cbad': cbad_evaluate_epoch,
    'dibco': dibco_evaluate_epoch,
    'cini': cini_evaluate_epoch
}


def _hash_dict(params):
    return sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()


def evaluate_one_model(model_dir, labels_dir, post_processing_eval_fn, post_processing_params,
                       verbose=False, save_params=True):
    eval_outputs_dir = os.path.join(model_dir, 'eval', 'epoch_*')
    list_saved_epochs = glob(eval_outputs_dir)
    if len(list_saved_epochs) == 0:
        print('No file found in : {}'.format(eval_outputs_dir))
        return

    post_process_dir = os.path.join(model_dir, POST_PROCESSING_DIR_NAME, _hash_dict(post_processing_params))
    os.makedirs(post_process_dir, exist_ok=True)

    validation_scores = dict()
    for saved_epoch in list_saved_epochs:
        epoch_dir_name = saved_epoch.split(os.path.sep)[-1]
        epoch, timestamp = (int(s) for s in epoch_dir_name.split('_')[1:3])
        validation_scores[epoch_dir_name] = {**post_processing_eval_fn(saved_epoch, labels_dir,
                                                                       verbose=verbose,
                                                                       **post_processing_params),
                                             "epoch": epoch,
                                             "timestamp": timestamp
                                             }

    with open(os.path.join(post_process_dir, 'validation_scores.json'), 'w') as f:
        json.dump(validation_scores, f)

    if save_params:
        with open(os.path.join(post_process_dir, 'post_process_params.json'), 'w') as f:
            json.dump(post_processing_params, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-dir', type=str, required=True, nargs='+')
    parser.add_argument('-l', '--labels-dir', type=str, required=True)
    parser.add_argument('-p', '--params-json-file', type=str, required=True)
    parser.add_argument('-t', '--task-type', type=str, required=True)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = vars(parser.parse_args())

    post_processing_eval_fn = POST_PROCESSING_EVAL_FN_DICT[args['task_type']]

    with open(args.get('params_json_file'), 'r') as f:
        configs_data = json.load(f)
        # If the file contains a list of configurations
        if 'configs' in configs_data.keys():
            params_list = configs_data['configs']
            assert isinstance(params_list, list)
        # Or if there is a single configuration
        else:
            params_list = [configs_data]

    model_dirs = args.get('model_dir')
    print('Found {} configs and {} model directories'.format(len(params_list), len(model_dirs)))

    for params in tqdm(params_list):
        for model_dir in tqdm(model_dirs):
            evaluate_one_model(model_dir, args.get('labels_dir'),
                               post_processing_eval_fn,
                               params, args.get('verbose'))
