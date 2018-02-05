#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
import json
from doc_seg_datasets.evaluation import dibco_evaluate_epoch
import argparse
import time


PARAMS_POST_PROCESSING = {'threshold': -1}
POST_PROCESSING_DIR_NAME = os.path.join('post_processing', '{}'.format(int(time.time())))


def evaluate_one_dibco_model(model_dir, labels_dir, post_processing_params, verbose=False, save_params=True):

    post_process_dir = os.path.join(model_dir, POST_PROCESSING_DIR_NAME)
    os.makedirs(post_process_dir, exist_ok=True)

    list_saved_epochs = glob(os.path.join(model_dir, 'exported_eval_files', '*'))

    measures = list()
    for saved_epoch in list_saved_epochs:
        epoch_dir_name = saved_epoch.split(os.path.sep)[-1]
        measures.append((epoch_dir_name,
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
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-v', '--labels_dir', type=str, required=True)
    parser.add_argument('-p', '--params_json_file', type=str, default=None)
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    args = vars(parser.parse_args())

    post_process_dir = os.path.join(args.get('model_dir'), POST_PROCESSING_DIR_NAME)

    if args.get('params_json_file') is None:
        os.makedirs(post_process_dir, exist_ok=True)
        params = PARAMS_POST_PROCESSING
        with open(os.path.join(post_process_dir, 'post_process_params.json'), 'w') as f:
            json.dump(PARAMS_POST_PROCESSING, f)
    else:
        with open(args.get('params_json_file'), 'r') as f:
            params = json.load(f)

    evaluate_one_dibco_model(args.get('model_dir'), args.get('labels_dir'), params,
                             args.get('verbose'), save_params=False)

