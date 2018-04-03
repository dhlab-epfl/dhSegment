#!/usr/bin/env python
import argparse
from typing import List
from doc_seg.evaluation.model_selection import ExperimentResult
from doc_seg.evaluation import dibco_evaluate_folder, cbad_evaluate_folder, \
    cini_evaluate_folder, page_evaluate_folder, diva_evaluate_folder, ornament_evaluate_folder
from doc_seg.loader import LoadedModel
from doc_seg import post_processing, utils
import tensorflow as tf
from glob import glob
import os
from tqdm import tqdm


POST_PROCESSING_EVAL_FN_DICT = {
    'cbad': cbad_evaluate_folder,
    'dibco': dibco_evaluate_folder,
    'cini': cini_evaluate_folder,
    'page': page_evaluate_folder,
    'diva': diva_evaluate_folder,
    'ornaments': ornament_evaluate_folder
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-dirs', type=str, required=True, nargs='+')
    parser.add_argument('-t', '--test-folder', type=str, required=True)
    parser.add_argument('-m', '--selection-metric', type=str, required=True)
    parser.add_argument('-o', '--output-folder', type=str, required=False)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='False to run model on set, True evaluate only')
    args = vars(parser.parse_args())

    experiment_dirs = args['experiment_dirs']
    selection_metric = args['selection_metric']
    output_folder = args['output_folder']

    experiments = []  # type: List[ExperimentResult]
    for f in experiment_dirs:
        experiments.append(ExperimentResult(f, selection_metric))
    experiments = [e for e in experiments if 'epoch' in e.get_best_validated_epoch()]

    print('Found {} experiments'.format(len(experiments)))

    sorted_experiments = sorted(experiments, key=lambda s: s.get_best_validated_score(), reverse=True)

    best_experiment = sorted_experiments[0]  # type: ExperimentResult
    print('Best selected experiment :')
    print(best_experiment)
    print('--------------------')
    print('Corresponding scores :')
    print(best_experiment.get_best_validated_epoch())

    # Perform test prediction (is it the right place?)
    test_folder = args.get('test_folder')
    for i, best_experiment in enumerate(sorted_experiments[:10]):
        print(best_experiment)
        print('Validation :')
        print(best_experiment.get_best_validated_epoch())
        model_folder = best_experiment.get_best_model_folder()
        post_process_fn = getattr(post_processing, best_experiment.post_process_config['post_process_fn'])
        post_process_params = best_experiment.post_process_config['params']
        output_folder_exp = os.path.join(output_folder, '{}'.format(i))
        os.makedirs(output_folder_exp, exist_ok=True)
        test_files = glob(os.path.join(test_folder, 'images', '*.jpg'))
        if len(test_files) == 0:
            print('No files in ', test_folder)
            continue
        if not args.get('eval_only'):
            with tf.Graph().as_default(), tf.Session() as sess:
                m = LoadedModel(model_folder, input_dict_key='filename')
                for filename in tqdm(test_files, desc='Test images'):
                    basename = os.path.basename(filename).split('.')[0]
                    probs = m.predict(filename, prediction_key='probs')[0]
                    output = post_process_fn(probs, **post_process_params,
                                             output_basename=os.path.join(output_folder_exp,
                                                                          os.path.splitext(os.path.basename(filename))[0]))

        print('Test :')
        scores = POST_PROCESSING_EVAL_FN_DICT[args.get('task')](output_folder_exp, test_folder,
                                                                debug_folder=os.path.join(output_folder_exp, 'debug'))
        utils.dump_json(os.path.join(output_folder_exp, 'scores.json'), scores)
        print(scores)
        utils.dump_json(os.path.join(output_folder_exp, 'experiment_info.json'),
                        {'model': best_experiment.model_config,
                         'post_process_config': best_experiment.post_process_config,
                         'folder': best_experiment.get_best_model_folder()})
