#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from tqdm import tqdm
from glob import glob
import os
from imageio import imsave
import subprocess
import tempfile
import numpy as np
from .process import diva_post_processing_fn
from .utils import to_original_color_code, parse_diva_tool_output


DIVA_JAR = './DIVA_Layout_Analysis_Evaluator/out/artifacts/LayoutAnalysisEvaluator.jar'
PP_PARAMS = {'thresholds': [0.5, 0.5, 0.5], 'min_cc': 50}


def eval_fn(input_dir: str, groundtruth_dir: str, output_filename: str, post_process_params: dict=PP_PARAMS,
            diva_jar: str=DIVA_JAR):
    """

    :param input_dir: directory containing the predictions .npy files (range [0, 255])
    :param groundtruth_dir: directory containing the ground truth images (.png) (must have the same name as predictions
                            files in input_dir)
    :param output_filename: filename of the .txt file containing all the results computed by the Evaluation tool
    :param post_process_params: params for post processing fn
    :param diva_jar: path for the DIVA Evaluation Tool (.jar file)
    :return: mean IU
    """
    results_list = list()
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in tqdm(glob(os.path.join(input_dir, '*.npy'))):
            basename = os.path.basename(file).split('.')[0]

            pred = np.load(file)
            pp_preds = diva_post_processing_fn(pred/np.max(pred), **post_process_params)

            original_colors = to_original_color_code(np.uint8(pp_preds * 255))
            pred_img_filename = os.path.join(tmpdir, '{}_orig_colors.png'.format(basename))
            imsave(pred_img_filename, original_colors)

            label_image_filename = os.path.join(groundtruth_dir, basename + '.png')

            cmd = 'java -jar {} -gt {} -p {}'.format(diva_jar, label_image_filename, pred_img_filename)
            result = subprocess.check_output(cmd, shell=True).decode()
            results_list.append(result)

    mius = list()
    with open(output_filename, 'w') as f:
        for result in results_list:
            r = parse_diva_tool_output(result)
            mius.append(r['Mean_IU'])
            f.write(result)
        f.write('--- Mean IU : {}'.format(np.mean(mius)))

    print('Mean IU : {}'.format(np.mean(mius)))
    return np.mean(mius)
