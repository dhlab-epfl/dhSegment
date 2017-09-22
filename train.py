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

    parameters_model = utils.Params(
        input_dir_train=args.get('train_dir'),
        input_dir_eval=args.get('eval_dir'),
        output_model_dir=args.get('model_output_dir'),
        n_epochs=args.get('nb_epochs'),
        gpu=args.get('gpu'),
        learning_rate=1e-5,  # 1e-5
        weight_decay=1e-4,  # 1e-5
        make_patches=False,
        vgg_intermediate_conv=[
            [(256, 3)]
        ],
        vgg_upscale_params=[
            [(64, 3)],
            [(128, 3)],
            [(256, 3)],
            [(512, 3)],
            [(512, 3)]
        ],
        vgg_selected_levels_upscaling=[True,  # Must have same length as vgg_upscale_params
                                       True,
                                       True,
                                       False,
                                       False],
        resized_size=(480, 320),  # (15,10)*32
        prediction_type=args.get('prediction_type'),
        class_file=args.get('classes_file'),
        model_name='vgg16',
        # pretrained_file='/mnt/cluster-nas/benoit/pretrained_nets/vgg_16.ckpt'
    )

    if parameters_model.prediction_type == utils.PredictionType.CLASSIFICATION:
        classes = utils.get_classes_color_from_file(args.get('classes_file'))
        parameters_model.n_classes = classes.shape[0]

    parameters_model.export_experiment_params()

    model_params = {'Params': parameters_model}

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = parameters_model.gpu
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10)
    estimator = tf.estimator.Estimator(model.model_fn, model_dir=parameters_model.output_model_dir,
                                       params=model_params, config=estimator_config)

    train_images_dir, train_labels_dir = os.path.join(parameters_model.input_dir_train, 'images'), \
                                         os.path.join(parameters_model.input_dir_train, 'labels')
    eval_images_dir, eval_labels_dir = os.path.join(parameters_model.input_dir_eval, 'images'), \
                                       os.path.join(parameters_model.input_dir_eval, 'labels')

    for i in trange(0, parameters_model.n_epochs, parameters_model.evaluate_every_epoch):
        # Train for one epoch
        estimator.train(input.input_fn(input_image_dir=train_images_dir,
                                       input_label_dir=train_labels_dir,
                                       num_epochs=parameters_model.evaluate_every_epoch,
                                       batch_size=parameters_model.batch_size,
                                       data_augmentation=parameters_model.data_augmentation,
                                       make_patches=parameters_model.make_patches,
                                       image_summaries=True,
                                       model_params=parameters_model))
        # Evaluate
        estimator.evaluate(input.input_fn(input_image_dir=eval_images_dir,
                                          input_label_dir=eval_labels_dir,
                                          num_epochs=1,
                                          batch_size=parameters_model.batch_size,
                                          model_params=parameters_model))

    # Exporting model
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'images': tf.placeholder(tf.float32, [None, None, None, 3])
    })
    estimator.export_savedmodel(os.path.join(parameters_model.output_model_dir, 'export'), export_input_fn)
