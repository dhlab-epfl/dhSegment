import json
import os
import argparse
import copy

params_training_size = [int(100e4), int(60e4)]
# params_training_size = [-1]
# params_training_learning_rate = [1e-4,
#                                  1e-3]
params_training_focal_loss_gamma = [0.0, 1.0]


def deep_update(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, dict())
            deep_update(node, value)
        else:
            destination[key] = value

    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--config-output-dir', type=str, required=True)
    parser.add_argument('-mo', '--model-output-dir', type=str, required=True)
    # One or possibly more base configurations
    parser.add_argument('-c', '--base-config', type=str, required=True, nargs='+')
    parser.add_argument('-t', '--task', type=str, default='')
    args = vars(parser.parse_args())

    os.makedirs(args['config_output_dir'], exist_ok=True)

    id = 0
    base_config = dict()
    for filename in args['base_config']:
        with open(filename, 'r') as f:
            deep_update(base_config, json.load(f))  # type: dict

    dic_params = copy.deepcopy(base_config)
    # for sel_levels in params_model_selected_levels:
    #     dic_params['model_params'] = dict()
    #     dic_params['model_params']['selected_levels_upscaling'] = sel_levels
    #     for upscale in params_model_upsale_params:
    #         dic_params['model_params']['upscale_params'] = upscale
    for size in params_training_size:
        dic_params['training_params']['input_resized_size'] = size
        for gamma in params_training_focal_loss_gamma:
            dic_params['training_params']['focal_loss_gamma'] = gamma
        # for learing_rate in params_training_learning_rate:
        #     dic_params['training_params']['learning_rate'] = learing_rate

            dic_params['model_output_dir'] = os.path.join(args.get('model_output_dir'),
                                                          'exp_{:03d}_{}'.format(id, args.get('task')))

            with open(os.path.join(args.get('config_output_dir'),
                                   '{}_variant_{:03d}.json'.format(args.get('task'), id)), 'w') as f:
                json.dump(dic_params, f)
            id += 1
    print('Generated {} configs'.format(id))
