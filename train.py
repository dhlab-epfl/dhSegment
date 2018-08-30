import os
import tensorflow as tf
# Tensorflow logging level
from logging import WARNING  # import  DEBUG, INFO, ERROR for more/less verbosity
tf.logging.set_verbosity(WARNING)
from dh_segment import estimator_fn, input, utils
import json
from glob import glob
import numpy as np
try:
    import better_exceptions
except ImportError:
    print('/!\ W -- Not able to import package better_exceptions')
    pass
from tqdm import trange
from sacred import Experiment
import pandas as pd

ex = Experiment('dhSegment_experiment')


@ex.config
def default_config():
    train_dir_or_csv = None  # Directory with training data
    eval_dir_or_csv = None  # Directory with validation data
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
def run(train_dir_or_csv, eval_dir_or_csv, model_output_dir, gpu, training_params, _config):

    # Create output directory
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)
    else:
        assert _config.get('restore_model'), \
            '{0} already exists, you cannot use it as output directory. ' \
            'Set "restore_model=True" to continue training, or delete dir "rm -r {0}"'.format(model_output_dir)
    # Save config
    with open(os.path.join(model_output_dir, 'config.json'), 'w') as f:
        json.dump(_config, f, indent=4, sort_keys=True)

    # Create export directory for saved models
    saved_model_dir = os.path.join(model_output_dir, 'export')
    if not os.path.isdir(saved_model_dir):
        os.makedirs(saved_model_dir)

    training_params = utils.TrainingParams.from_dict(training_params)

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = str(gpu)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10,
                                                        keep_checkpoint_max=1)
    estimator = tf.estimator.Estimator(estimator_fn.model_fn, model_dir=model_output_dir,
                                       params=_config, config=estimator_config)

    if os.path.isdir(train_dir_or_csv):
        train_input, train_labels_input = os.path.join(train_dir_or_csv, 'images'), os.path.join(train_dir_or_csv, 'labels')
        # Check if training dir exists
        assert os.path.isdir(train_input)
        if eval_dir_or_csv is not None:
            eval_images_dir, eval_labels_dir = os.path.join(eval_dir_or_csv, 'images'), os.path.join(eval_dir_or_csv, 'labels')
            assert os.path.isdir(eval_images_dir)
            filenames_evaluation = glob(os.path.join(eval_images_dir, '*.jpg')) \
                                   + glob(os.path.join(eval_images_dir, '*.png'))
    elif os.path.isfile(train_dir_or_csv):
        train_input = train_dir_or_csv
        train_labels_input = None
        eval_input = eval_dir_or_csv
        eval_labels_input = None
        # Get filenames for evaluation
        df = pd.read_csv(eval_dir_or_csv, header=None, names=['images', 'labels'])
        filenames_evaluation = list(df.images)
    else:
        raise TypeError('train_dir_or_csv is neither a directory yor a csv file')

    for i in trange(0, training_params.n_epochs, training_params.evaluate_every_epoch, desc='Evaluated epochs'):
        estimator.train(input.input_fn(train_input,
                                       input_label_dir=train_labels_input,
                                       num_epochs=training_params.evaluate_every_epoch,
                                       batch_size=training_params.batch_size,
                                       data_augmentation=training_params.data_augmentation,
                                       make_patches=training_params.make_patches,
                                       image_summaries=True,
                                       params=_config,
                                       num_threads=32))

        # Export model (filename, batch_size = 1) and predictions
        exported_path = estimator.export_savedmodel(saved_model_dir,
                                                    input.serving_input_filename(training_params.input_resized_size))
        exported_path = exported_path.decode()
        timestamp_exported = os.path.split(exported_path)[-1]

        if eval_dir_or_csv is not None:
            try:  # There should be no evaluation when input_resize_size is too big (e.g -1)
                # Save predictions
                exported_files_eval_dir = os.path.join(model_output_dir, 'eval',
                                                       'epoch_{:03d}_{}'.format(i+training_params.evaluate_every_epoch,
                                                                                timestamp_exported))
                os.makedirs(exported_files_eval_dir, exist_ok=True)
                # Predict and save probs
                prediction_input_fn = input.input_fn(filenames_evaluation, num_epochs=1, batch_size=1,
                                                     data_augmentation=False, make_patches=False, params=_config,
                                                     num_threads=32)
                for filename, predicted_probs in zip(filenames_evaluation,
                                                     estimator.predict(prediction_input_fn, predict_keys=['probs'])):
                    np.save(os.path.join(exported_files_eval_dir, os.path.basename(filename).split('.')[0]),
                            np.uint8(255 * predicted_probs['probs']))
                # estimator.evaluate(input.input_fn(eval_input,
                #                                   input_label_dir=eval_labels_input,
                #                                   num_epochs=1,
                #                                   batch_size=training_params.batch_size,
                #                                   params=_config,
                #                                   num_threads=32))
            except Exception as e:
                print(e)
