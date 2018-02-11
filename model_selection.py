#!/usr/bin/env python
import argparse
import json
from typing import List
from doc_seg.evaluation.model_selection import ExperimentResult


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-dirs', type=str, required=True, nargs='+')
    parser.add_argument('-m', '--selection-metric', type=str, required=True)
    args = vars(parser.parse_args())

    experiment_dirs = args['experiment_dirs']
    selection_metric = args['selection_metric']

    experiments = []  # type: List[ExperimentResult]
    for f in experiment_dirs:
        experiments.append(ExperimentResult(f, selection_metric))
    experiments = [e for e in experiments if 'epoch' in e.get_best_validated_epoch()]

    print('Found {} experiments'.format(len(experiments)))

    sorted_experiments = sorted(experiments, key=lambda s: s.get_best_validated_score(), reverse=True)

    print('Best selected experiment :')
    print(sorted_experiments[0])
    print('--------------------')
    print('Corresponding scores :')
    print(sorted_experiments[0].get_best_validated_epoch())
