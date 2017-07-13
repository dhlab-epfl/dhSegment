import argparse
import tensorflow as tf
from doc_seg import model, input
import os
import better_exceptions
from tqdm import trange


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train-dir", required=True, help="Folder with training and label images")
ap.add_argument("-e", "--eval-dir", required=True, help="Folder with the evaluation images and labels")
ap.add_argument("-o", "--model-output-dir", required=True, help="Where the model will be saved")
ap.add_argument("--nb-epochs", default=20, type=int, help="Number of epochs")
args = vars(ap.parse_args())


model_params = {
    'learning_rate': 1e-5,
    'num_classes': 3,
    'model_params': [
        [(32, 7), (32, 5)],
        [(64, 5), (64, 5)],
        [(128, 5), (128, 5)],
        [(128, 5), (128, 5)],
        [(128, 5), (128, 5)]
    ]
}

session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = '0'
session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                    save_summary_steps=10)
estimator = tf.estimator.Estimator(model.model_fn, model_dir=args['model_output_dir'],
                                   params=model_params, config=estimator_config)

train_images_dir, train_labels_dir = os.path.join(args['train_dir'], 'images'), os.path.join(args['train_dir'], 'labels')
eval_images_dir, eval_labels_dir = os.path.join(args['eval_dir'], 'images'), os.path.join(args['eval_dir'], 'labels')
for i in trange(args['nb_epochs']):
    # Train for one epoch
    estimator.train(input.input_fn(train_images_dir, train_labels_dir, num_epochs=1,
                                   data_augmentation=True, image_summaries=True))
    # Evaluate
    estimator.evaluate(input.input_fn(eval_images_dir, eval_labels_dir, num_epochs=1))

# Exporting model
estimator.export_savedmodel(args['model_output_dir']+'/export',
                            tf.estimator.export.build_raw_serving_input_receiver_fn({'images':
                                                                                     tf.placeholder(tf.float32,
                                                                                                    [None, 688, 1024, 3])}))
