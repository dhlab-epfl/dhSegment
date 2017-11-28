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
from sacred import Experiment

ex = Experiment('DocumentSegmentation_experiment')


@ex.config
def default_config():
    train_dir = None  # Directory with training data
    eval_dir = None  # Directory with validation data
    model_output_dir = None  # Directory to output tf model
    restore_model = False  # Set to true to continue training
    classes_file = None  # txt file with classes values (unused for REGRESSION)
    gpu = ''  # GPU to be used for training
    prediction_type = utils.PredictionType.CLASSIFICATION  # One of CLASSIFICATION, REGRESSION or MULTILABEL
    pretrained_model_name = 'resnet50'
    model_params = utils.ModelParams(pretrained_model_name=pretrained_model_name).to_dict()  # Model parameters
    training_params = utils.TrainingParams().to_dict()  # Training parameters
    if prediction_type == utils.PredictionType.CLASSIFICATION:
        assert classes_file is not None
        model_params['n_classes'] = utils.get_n_classes_from_file(classes_file)
    elif prediction_type == utils.PredictionType.REGRESSION:
        model_params['n_classes'] = 1
    elif prediction_type == utils.PredictionType.MULTILABEL:
        assert classes_file is not None
        model_params['n_classes'] = utils.get_n_classes_from_file_multilabel(classes_file)


@ex.automain
def run(train_dir, eval_dir, model_output_dir, gpu, training_params, _config):
    # Save config
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)
    else:
        assert _config.get('restore_model'), \
            '{} already exists, you cannot use it as output directory. ' \
            'Set "restore_model=True" to continue training'.format(model_output_dir)
    with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
        json.dump(_config, f)

    training_params = utils.TrainingParams.from_dict(training_params)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = str(gpu)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10)
    estimator = tf.estimator.Estimator(model.model_fn, model_dir=model_output_dir,
                                       params=_config, config=estimator_config)

    train_images_dir, train_labels_dir = os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels')
    eval_images_dir, eval_labels_dir = os.path.join(eval_dir, 'images'), os.path.join(eval_dir, 'labels')

    for i in trange(0, training_params.n_epochs, training_params.evaluate_every_epoch):
        # Train for one epoch
        estimator.train(input.input_fn(input_image_dir=train_images_dir,
                                       input_label_dir=train_labels_dir,
                                       num_epochs=training_params.evaluate_every_epoch,
                                       batch_size=training_params.batch_size,
                                       data_augmentation=training_params.data_augmentation,
                                       make_patches=training_params.make_patches,
                                       image_summaries=True,
                                       params=_config))
        # Evaluate
        estimator.evaluate(input.input_fn(input_image_dir=eval_images_dir,
                                          input_label_dir=eval_labels_dir,
                                          num_epochs=1,
                                          batch_size=training_params.batch_size,
                                          data_augmentation=False,
                                          make_patches=training_params.make_patches,
                                          params=_config))

    # Exporting model
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'images': tf.placeholder(tf.float32, [None, None, None, 3])
    })
    estimator.export_savedmodel(os.path.join(model_output_dir, 'export'), export_input_fn)

    # Update config
    with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
        json.dump(_config, f)
