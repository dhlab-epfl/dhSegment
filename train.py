import tensorflow as tf
from doc_seg import model, input, utils, loader
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
try:
    import better_exceptions
except:
    print('/!\ W -- Not able to import package better_exceptions')
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
        json.dump(_config, f, indent=4)

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

        # Export model (filename, batch_size = 1) and predictions
        _ = estimator.export_savedmodel(os.path.join(model_output_dir, 'export'), input.serving_input_filename())

        # Save predictions
        filenames_evaluation = glob(os.path.join(eval_images_dir, '*.jpg'))
        exported_files_eval_dir = os.path.join(model_output_dir, 'exported_eval_files', 'epoch_{}'.format(i))
        os.makedirs(exported_files_eval_dir, exist_ok=True)
        # Predict and save probs
        for filename in filenames_evaluation:  # tqdm(filenames_evaluation):
            predicted_probs = estimator.predict(input.prediction_input_filename(filename), predict_keys=['probs'])
            np.save(os.path.join(exported_files_eval_dir, os.path.basename(filename).split('.')[0]),
                    np.uint8(255 * next(predicted_probs)['probs']))

    # Exporting model
    estimator.export_savedmodel(os.path.join(model_output_dir, 'export'), input.serving_input_filename())
