import json
import os
import argparse
import copy

params_pretrained_model_name = ['resnet50', 'vgg16']
params_model_selected_levels = [
    # [True, True, True, True, True, True],
    [True, True, True, True, True, False],
    [True, True, True, True, False, False],
    [True, True, True, False, False, False],
    [True, True, False, False, False, False]

]
params_model_upsale_params = [
    [[(32, 3)], [(64, 3)], [(64, 3)], [(128, 1)], [(128, 1)], [(256, 1)]],
    [[(32, 3)], [(32, 3)], [(64, 3)], [(128, 1)], [(128, 1)], [(256, 1)]],
]

params_training_size = [int(82e4), int(72e4), int(60e4)]
params_training_make_patches = [True, False]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-co', '--config-output-dir', type=str, required=True)
    parser.add_argument('-mo', '--model-output-dir', type=str, required=True)
    # One or possibly more base configurations
    parser.add_argument('-c', '--base-config', type=str, required=True, nargs='+')
    args = vars(parser.parse_args())

    os.makedirs(args['config_output_dir'], exist_ok=True)

    id = 0
    base_config = dict()
    for filename in args['base_config']:
        with open(filename, 'r') as f:
            base_config.update(json.load(f))  # type: dict
    for pre_name in params_pretrained_model_name:
        dic_params = copy.deepcopy(base_config)
        dic_params['pretrained_model_name'] = pre_name
        for sel_levels in params_model_selected_levels:
            dic_params['model_params'] = dict()
            dic_params['model_params']['selected_levels_upscaling'] = sel_levels
            for upscale in params_model_upsale_params:
                dic_params['model_params']['upscale_params'] = upscale
                for size in params_training_size:
                    dic_params['training_params']['input_resized_size'] = size
                    for mp in params_training_make_patches:
                        dic_params['training_params']['make_patches'] = mp
                        dic_params['training_params']['batch_size'] = 3 if mp is False else 32
                    dic_params['model_output_dir'] = os.path.join(args.get('model_output_dir'), 'exp_{:03d}'.format(id))
                    with open(os.path.join(args.get('config_output_dir'), 'variant_{:03d}.json'.format(id)), 'w') as f:
                        json.dump(dic_params, f)
                    id += 1
    print('Generated {} configs'.format(id))
