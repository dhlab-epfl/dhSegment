import argparse
import tensorflow as tf
from doc_seg import model, input, utils
import os
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
        'learning_rate': 1e-5,
        # 'num_classes': 1, # by default
        #TODO batch_norm usage
        'model_params': [
            [(32, 7), (32, 5)],
            [(64, 5), (64, 5)],
            [(128, 5), (128, 5)],
            [(128, 5), (128, 5)],
            [(128, 5), (128, 5)]
        ],
        'prediction_type': prediction_type,
        'classes_file': args.get('classes_file')
    }
    if model_params['prediction_type'] == utils.PredictionType.CLASSIFICATION:
        assert model_params['classes_file'] is not None
        classes = utils.get_classes_color_from_file(args.get('classes_file'))
        model_params['num_classes'] = classes.shape[0]
    elif model_params['prediction_type'] == utils.PredictionType.REGRESSION:
        model_params['num_classes'] = 1

    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = args.get('gpu')
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        save_summary_steps=10)
    estimator = tf.estimator.Estimator(model.model_fn, model_dir=args['model_output_dir'],
                                       params=model_params, config=estimator_config)

    train_images_dir, train_labels_dir = os.path.join(args['train_dir'], 'images'), os.path.join(args['train_dir'], 'labels')
    eval_images_dir, eval_labels_dir = os.path.join(args['eval_dir'], 'images'), os.path.join(args['eval_dir'], 'labels')
    for i in trange(0, args['nb_epochs'], evaluate_every_epochs):
        # Train for one epoch
        estimator.train(input.input_fn(prediction_type=model_params['prediction_type'],
                                       input_folder=train_images_dir,
                                       label_images_folder=train_labels_dir,
                                       classes_file=model_params['classes_file'], num_epochs=evaluate_every_epochs,
                                       data_augmentation=True, image_summaries=True))
        # Evaluate
        estimator.evaluate(input.input_fn(prediction_type=model_params['prediction_type'],
                                          input_folder=eval_images_dir,
                                          label_images_folder=eval_labels_dir,
                                          classes_file=model_params['classes_file'], num_epochs=1))

    # Exporting model
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'images': tf.placeholder(tf.float32, [None, None, None, 3])
    })
    estimator.export_savedmodel(args['model_output_dir']+'/export', export_input_fn)
