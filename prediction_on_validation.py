#!/usr/bin/env python
__author__ = 'solivr'


import tensorflow as tf
from doc_seg import model, input
from tqdm import tqdm
import os
import argparse
from glob import glob
import numpy as np
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment_dirs', type=str, required=True, nargs='+')
    parser.add_argument('-v', '--validation_folder', type=str, required=True)
    args = vars(parser.parse_args())

    validation_files = glob(os.path.join(args.get('validation_folder'), 'images', '**'))
    model_dirs = args.get('experiment_dirs')
    for model_dir in tqdm(model_dirs, desc='Model dirs'):
        exported_models = glob(os.path.join(model_dir, 'export', '**'))
        exported_models.sort()
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        # for i, exported_model_folder in tqdm(enumerate(exported_models), desc='Exported'):

        estimator = tf.estimator.Estimator(model.model_fn, model_dir=model_dir,
                                           params=config)

        # timestamp = exported_model_folder.split('/')[-1]
        exported_files_eval_dir = os.path.join(model_dir, 'eval', 'epoch_000_000')
        os.makedirs(exported_files_eval_dir, exist_ok=True)

        prediction_input_fn = input.input_fn(validation_files, num_epochs=1, batch_size=1,
                                             data_augmentation=False, make_patches=False, params=config)

        for filename, predicted_probs in zip(validation_files,
                                             estimator.predict(prediction_input_fn, predict_keys=['probs'])):
            np.save(os.path.join(exported_files_eval_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * predicted_probs['probs']))