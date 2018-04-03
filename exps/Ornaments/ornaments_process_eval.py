#!/usr/bin/env python
__author__ = 'solivr'

import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from glob import glob
from ornaments_evaluation import ornament_evaluate_folder
from ornaments_post_processing import ornaments_post_processing_fn
import tempfile


PARAMS = {"threshold": 0.6, "ksize_open": [0, 0], "ksize_close": [0, 0]}
MIOU_THRESHOD = 0.8
MIN_AREA = 0.005

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--npy_directory', type=str, required=True,
                        help='Directory containing bin .png files')
    parser.add_argument('-gt', '--gt_directory', type=str, required=True,
                        help='Directory containing images and labels for evaluation')
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='Output directory')
    parser.add_argument('-p', '--params_file', type=str, required=False,
                        help='JSON params file')
    args = parser.parse_args()
    args = vars(args)

    output_dir = args.get('output_directory')
    npy_dir = args.get('npy_directory')
    # os.makedirs(output_dir)

    if args.get('params_file') is None:
        print('No params file found')
        params_list = [PARAMS]
    else:
        with open(args.get('params_file'), 'r') as f:
            configs_data = json.load(f)
            # If the file contains a list of configurations
            if 'configs' in configs_data.keys():
                params_list = configs_data['configs']
                assert isinstance(params_list, list)
            # Or if there is a single configuration
            else:
                params_list = [configs_data]

    npy_files = glob(os.path.join(npy_dir, '*.npy'))
    for params in params_list:
        new_output_dir = output_dir + 'th{}_a{}_{}'.format(MIOU_THRESHOD, MIN_AREA, np.random.randint(0, 1000))
        os.makedirs(new_output_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename in tqdm(npy_files):
                probs = np.load(filename)
                _ = ornaments_post_processing_fn(probs/np.max(probs), **params,
                                                 output_basename=os.path.join(tmpdir,
                                                                              os.path.basename(filename).split('.')[0]))

            measures = ornament_evaluate_folder(tmpdir, args.get('gt_directory'), min_area=MIN_AREA,
                                                miou_threshold=MIOU_THRESHOD, debug_folder=new_output_dir)

        with open(os.path.join(new_output_dir, 'validation_scores.json'), 'w') as f:
            json.dump(measures, f)
        with open(os.path.join(new_output_dir, 'post_process_params.json'), 'w') as f:
            json.dump(params, f)
