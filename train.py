import argparse
import tensorflow as tf
from doc_seg import model, input, utils
import os
import json
try:
    import better_exceptions
except:
    pass
from tqdm import trange


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train-dir", required=True, help="Folder with training and label images")
    ap.add_argument("-e", "--eval-dir", required=True, help="Folder with the evaluation images and labels")
    ap.add_argument("-o", "--model-output-dir", required=True, help="Where the model will be saved")
    ap.add_argument("-c", "--classes-file", required=False, help="Text file describing the classes (only for classification)")
    ap.add_argument("--nb-epochs", default=20, type=int, help="Number of epochs")
    ap.add_argument("-g", "--gpu", required=True, type=str, help='GPU 0, 1 or CPU ('') ')
    ap.add_argument("-p", "--prediction-type", required=True, type=str, help='CLASSIFICATION or REGRESSION')
    args = vars(ap.parse_args())

    prediction_type = utils.PredictionType.CLASSIFICATION if args.get('prediction_type') == 'CLASSIFICATION' else \
        utils.PredictionType.REGRESSION

    evaluate_every_epochs = 5
    model_params = {
        'learning_rate': 1e-5,  # 1e-5
        'exponential_learning': True,
        'batch_norm': True,
        'weight_decay': 1e-5,
        # TODO : put this in a config file
        # 'model_params': [
        #     [(32, 7), (32, 5)],
        #     [(64, 5), (64, 5)],
        #     [(128, 5), (128, 5)],
        #     [(128, 5), (128, 5)],
        #     [(128, 5), (128, 5)]
        # ],
        # 'vgg_conv_params': [(64, 1)],
        'vgg_upscale_params': [
            [(64, 3)],
            [(128, 3)],
            [(256, 3)],
            [(512, 3)],
            [(512, 3)]
        ],
        'vgg_selected_levels_upscaling': [True,  # Must have same length as vgg_upscale_params
                                          True,
                                          True,
                                          True,
                                          True],
        'resized_size': (480, 320),  # (15,10)*32
        'prediction_type': prediction_type,
        'classes_file': args.get('classes_file'),
        'pretrained_file': '/mnt/cluster-nas/benoit/pretrained_nets/vgg_16.ckpt'
    }
    if model_params['prediction_type'] == utils.PredictionType.CLASSIFICATION:
        assert model_params['classes_file'] is not None
        classes = utils.get_classes_color_from_file(args.get('classes_file'))
        model_params['num_classes'] = classes.shape[0]
    elif model_params['prediction_type'] == utils.PredictionType.REGRESSION:
        model_params['num_classes'] = 1

    assert len(model_params['vgg_upscale_params']) == len(model_params['vgg_selected_levels_upscaling']), \
        'Upscaling levels and selection levels must have the same lengths (in model_params definition), ' \
        '{} != {}'.format(len(model_params['vgg_upscale_params']), len(model_params['vgg_selected_levels_upscaling']))

    # Exporting params
    if not os.path.isdir(args['model_output_dir']):
        os.mkdir(args['model_output_dir'])
    with open(os.path.join(args['model_output_dir'], 'model_params.json'), 'w') as f:
        json.dump(model_params, f)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = args.get('gpu')
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10)
    estimator = tf.estimator.Estimator(model.model_fn, model_dir=args['model_output_dir'],
                                       params=model_params, config=estimator_config)

    train_images_dir, train_labels_dir = os.path.join(args['train_dir'], 'images'), \
                                         os.path.join(args['train_dir'], 'labels')
    eval_images_dir, eval_labels_dir = os.path.join(args['eval_dir'], 'images'), \
                                       os.path.join(args['eval_dir'], 'labels')
    input_fn_args = dict(prediction_type=model_params['prediction_type'],
                         classes_file=model_params['classes_file'],
                         resized_size=model_params['resized_size']
                         )
    for i in trange(0, args['nb_epochs'], evaluate_every_epochs):
        # Train for one epoch
        estimator.train(input.input_fn(input_folder=train_images_dir,
                                       label_images_folder=train_labels_dir,
                                       num_epochs=evaluate_every_epochs,
                                       data_augmentation=True, image_summaries=True,
                                       **input_fn_args))
        # Evaluate
        estimator.evaluate(input.input_fn(input_folder=eval_images_dir,
                                          label_images_folder=eval_labels_dir,
                                          num_epochs=1, **input_fn_args))

    # Exporting model
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'images': tf.placeholder(tf.float32, [None, None, None, 3])
    })
    estimator.export_savedmodel(args['model_output_dir']+'/export', export_input_fn)
