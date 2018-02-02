#!/usr/bin/env python
__author__ = 'solivr'

import os
from glob import glob
import json
from doc_seg_datasets.evaluation import dibco_evaluate_epoch
import argparse
import time


PARAMS_POST_PROCESSING = {'threshold': -1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, required=True)
    parser.add_argument('-v', '--labels_dir', type=str, required=True)
    parser.add_argument('-p', '--params_json_file', type=str, default=None)
    args = vars(parser.parse_args())

    post_process_dir = os.path.join(args.get('model_dir'), 'post_processing', '{}'.format(int(time.time())))
    os.makedirs(post_process_dir, exist_ok=True)

    saved_epochs_full_path = glob(os.path.join(args.get('model_dir'), 'exported_eval_files', '*'))

    if args.get('params_json_file') is None:
        params = PARAMS_POST_PROCESSING
        with open(os.path.join(post_process_dir, 'post_process_params.json'), 'w') as f:
            json.dump(PARAMS_POST_PROCESSING, f)
    else:
        with open(args.get('params_json_file'), 'r') as f:
            params = json.load(f)

    measures = list()
    for saved_epoch in saved_epochs_full_path:
        epoch_dir_name = saved_epoch.split(os.path.sep)[-1]
        measures.append((epoch_dir_name,
                         dibco_evaluate_epoch(saved_epoch, args.get('labels_dir'),
                                              verbose=True, threshold=params['threshold'])))

    validation_scores = {epoch: {k: v for k, v in vars(metric).items()
                                 if k in ['MSE', 'PSNR', 'precision', 'recall', 'f_measure']}
                         for (epoch, metric) in measures}

    with open(os.path.join(post_process_dir, 'validation_scores.json'), 'w') as f:
        json.dump(validation_scores, f)

