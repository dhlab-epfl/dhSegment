import json
import os
import argparse

CONFIG_DIR = '/scratch/docseg_models/configs/'
TRAIN_DIR = '/scratch/dataset/dibco/generated_dibco/train/'
CLASSES_FILE = '/scratch/dataset/dibco/generated_dibco/classes.txt'
EVAL_DIR = '/scratch/dataset/dibco/generated_dibco/validation/'
MODEL_DIR = '/scratch/docseg_models/DIBCO/'
PREDICTION_TYPE = 'CLASSIFICATION'

params_pretrained_model_name = ['resnet50', 'vgg16']
params_model_selected_levels = [[True, True, True, True, False], [True, True, True, False, False],
                                [True, True, False, False, False]]
params_model_upsale_params = [[[(64, 3)], [(64, 3)], [(128, 3)], [(256, 3)], [(512, 3)]],
                              [[(64, 3)], [(128, 3)], [(256, 3)], [(512, 3)], [(512, 3)]],
                              [[(64, 1)], [(64, 1)], [(128, 1)], [(256, 1)], [(512, 1)]],
                              [[(64, 1)], [(128, 1)], [(256, 1)], [(512, 1)], [(512, 1)]]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--config_output_dir', type=str, default=CONFIG_DIR)
    parser.add_argument('-mo', '--model_output_dir', type=str, default=MODEL_DIR)
    parser.add_argument('-t', '--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('-ev', '--eval_dir', type=str, default=EVAL_DIR)
    parser.add_argument('-class', '--class_filename', type=str, default=CLASSES_FILE)
    parser.add_argument('-p', '--prediction_type', type=str, default=PREDICTION_TYPE)
    args = vars(parser.parse_args())

    id = 0
    for pre_name in params_pretrained_model_name:
        dic_params = dict()
        dic_params['pretrained_model_name'] = pre_name
        dic_params['train_dir'] = args.get('train_dir')
        dic_params['eval_dir'] = args.get('eval_dir')
        dic_params['classes_file'] = args.get('class_filename')
        dic_params['prediction_type'] = args.get('prediction')
        for sel_levels in params_model_selected_levels:
            dic_params['model_params'] = dict()
            dic_params['model_params']['selected_levels_upscaling'] = sel_levels
            for upscale in params_model_upsale_params:
                dic_params['model_params']['upscale_params'] = upscale
                dic_params['model_output_dir'] = os.path.join(args.get('model_output_dir'), 'exp_{}'.format(id))
                with open(os.path.join(args.get('config_output_dir'), 'variant_{}.json'.format(id)), 'w') as f:
                    json.dump(dic_params, f)
                id += 1
