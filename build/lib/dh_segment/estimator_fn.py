import tensorflow as tf
from .utils import PredictionType, ModelParams, TrainingParams, \
    class_to_label_image, multiclass_to_label_image
import numpy as np
from .network.model import inference_resnet_v1_50, inference_vgg16, inference_u_net


def model_fn(mode, features, labels, params):
    model_params = ModelParams(**params['model_params'])
    training_params = TrainingParams.from_dict(params['training_params'])
    prediction_type = params['prediction_type']
    classes_file = params['classes_file']

    input_images = features['images']

    if mode == tf.estimator.ModeKeys.PREDICT:
        margin = training_params.training_margin
        input_images = tf.pad(input_images, [[0, 0], [margin, margin], [margin, margin], [0, 0]],
                              mode='SYMMETRIC', name='mirror_padding')

    if model_params.pretrained_model_name == 'vgg16':
        network_output = inference_vgg16(input_images,
                                         model_params,
                                         model_params.n_classes,
                                         use_batch_norm=model_params.batch_norm,
                                         weight_decay=model_params.weight_decay,
                                         is_training=(mode == tf.estimator.ModeKeys.TRAIN)
                                         )
        key_restore_model = 'vgg_16'

    elif model_params.pretrained_model_name == 'resnet50':
        network_output = inference_resnet_v1_50(input_images,
                                                model_params,
                                                model_params.n_classes,
                                                use_batch_norm=model_params.batch_norm,
                                                weight_decay=model_params.weight_decay,
                                                is_training=(mode == tf.estimator.ModeKeys.TRAIN)
                                                )
        key_restore_model = 'resnet_v1_50'
    elif model_params.pretrained_model_name == 'unet':
        network_output = inference_u_net(input_images,
                                         model_params,
                                         model_params.n_classes,
                                         use_batch_norm=model_params.batch_norm,
                                         weight_decay=model_params.weight_decay,
                                         is_training=(mode == tf.estimator.ModeKeys.TRAIN)
                                         )
        key_restore_model = None
    else:
        raise NotImplementedError

    if mode == tf.estimator.ModeKeys.TRAIN:
        if key_restore_model is not None:
            # Pretrained weights as initialization
            pretrained_restorer = tf.train.Saver(var_list=[v for v in tf.global_variables()
                                                           if key_restore_model in v.name])

            def init_fn(scaffold, session):
                pretrained_restorer.restore(session, model_params.pretrained_model_file)
        else:
            init_fn = None
    else:
        init_fn = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        margin = training_params.training_margin
        # Crop padding
        if margin > 0:
            network_output = network_output[:, margin:-margin, margin:-margin, :]

    # Prediction
    # ----------
    if prediction_type == PredictionType.CLASSIFICATION:
        prediction_probs = tf.nn.softmax(network_output, name='softmax')
        prediction_labels = tf.argmax(network_output, axis=-1, name='label_preds')
        predictions = {'probs': prediction_probs, 'labels': prediction_labels}
    elif prediction_type == PredictionType.REGRESSION:
        predictions = {'output_values': network_output}
        prediction_labels = network_output
    elif prediction_type == PredictionType.MULTILABEL:
        with tf.name_scope('prediction_ops'):
            prediction_probs = tf.nn.sigmoid(network_output, name='sigmoid')  # [B,H,W,C]
            prediction_labels = tf.cast(tf.greater_equal(prediction_probs, 0.5, name='labels'), tf.int32)  # [B,H,W,C]
            predictions = {'probs': prediction_probs, 'labels': prediction_labels}
    else:
        raise NotImplementedError

    # Loss
    # ----
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        regularized_loss = tf.losses.get_regularization_loss()
        if prediction_type == PredictionType.CLASSIFICATION:
            onehot_labels = tf.one_hot(indices=labels, depth=model_params.n_classes)
            with tf.name_scope("loss"):
                per_pixel_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network_output,
                                                                         labels=onehot_labels, name='per_pixel_loss')
                if training_params.focal_loss_gamma > 0.0:
                    # Probability per pixel of getting the correct label
                    probs_correct_label = tf.reduce_max(tf.multiply(prediction_probs, onehot_labels))
                    modulation = tf.pow((1. - probs_correct_label), training_params.focal_loss_gamma)
                    per_pixel_loss = tf.multiply(per_pixel_loss, modulation)

                if training_params.weights_labels is not None:
                    weight_mask = tf.reduce_sum(
                        tf.constant(np.array(training_params.weights_labels, dtype=np.float32)[None, None, None]) *
                        onehot_labels, axis=-1)
                    per_pixel_loss = per_pixel_loss * weight_mask
                if training_params.local_entropy_ratio > 0:
                    assert 'weight_maps' in features
                    r = training_params.local_entropy_ratio
                    per_pixel_loss = per_pixel_loss * ((1 - r) + r * features['weight_maps'])

        elif prediction_type == PredictionType.REGRESSION:
            per_pixel_loss = tf.squared_difference(labels, network_output, name='per_pixel_loss')
        elif prediction_type == PredictionType.MULTILABEL:
            with tf.name_scope('sigmoid_xentropy_loss'):
                labels_floats = tf.cast(labels, tf.float32)
                per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats,
                                                                         logits=network_output, name='per_pixel_loss')
                if training_params.weights_labels is not None:
                    weight_mask = tf.maximum(
                        tf.reduce_max(tf.constant(
                            np.array(training_params.weights_labels, dtype=np.float32)[None, None, None])
                                      * labels_floats, axis=-1), 1.0)
                    per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]
        else:
            raise NotImplementedError

        margin = training_params.training_margin
        input_shapes = features['shapes']
        with tf.name_scope('Loss'):
            def _fn(_in):
                output, shape = _in
                return tf.reduce_mean(output[margin:shape[0] - margin, margin:shape[1] - margin])

            per_img_loss = tf.map_fn(_fn, (per_pixel_loss, input_shapes), dtype=tf.float32)
            loss = tf.reduce_mean(per_img_loss, name='loss')

        loss += regularized_loss
    else:
        loss, regularized_loss = None, None

    # Train
    # -----
    if mode == tf.estimator.ModeKeys.TRAIN:
        # >> Stucks the training... Why ?
        # ema = tf.train.ExponentialMovingAverage(0.9)
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        # ema_loss = ema.average(loss)

        if training_params.exponential_learning:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(training_params.learning_rate, global_step, decay_steps=200,
                                                       decay_rate=0.95, staircase=False)
        else:
            learning_rate = training_params.learning_rate
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        ema_loss, train_op = None, None

    # Summaries
    # ---------
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope('summaries'):
            tf.summary.scalar('losses/loss', loss)
            tf.summary.scalar('losses/loss_per_batch', loss)
            tf.summary.scalar('losses/regularized_loss', regularized_loss)
            if prediction_type == PredictionType.CLASSIFICATION:
                tf.summary.image('output/prediction',
                                 tf.image.resize_images(class_to_label_image(prediction_labels, classes_file),
                                                        tf.cast(tf.shape(network_output)[1:3] / 3, tf.int32)),
                                 max_outputs=1)
                if model_params.n_classes == 3:
                    tf.summary.image('output/probs',
                                     tf.image.resize_images(prediction_probs[:, :, :, :],
                                                            tf.cast(tf.shape(network_output)[1:3] / 3, tf.int32)),
                                     max_outputs=1)
                if model_params.n_classes == 2:
                    tf.summary.image('output/probs',
                                     tf.image.resize_images(prediction_probs[:, :, :, 1:2],
                                                            tf.cast(tf.shape(network_output)[1:3] / 3, tf.int32)),
                                     max_outputs=1)
            elif prediction_type == PredictionType.REGRESSION:
                summary_img = tf.nn.relu(network_output)[:, :, :, 0:1]  # Put negative values to zero
                tf.summary.image('output/prediction', summary_img, max_outputs=1)
            elif prediction_type == PredictionType.MULTILABEL:
                labels_visualization = tf.cast(prediction_labels, tf.int32)
                labels_visualization = multiclass_to_label_image(labels_visualization, classes_file)
                tf.summary.image('output/prediction_image',
                                 tf.image.resize_images(labels_visualization,
                                                        tf.cast(tf.shape(labels_visualization)[1:3] / 3, tf.int32)),
                                 max_outputs=1)
                class_dim = prediction_probs.get_shape().as_list()[-1]
                for c in range(0, class_dim):
                    tf.summary.image('output/prediction_probs_{}'.format(c),
                                     tf.image.resize_images(prediction_probs[:, :, :, c:c + 1],
                                                            tf.cast(tf.shape(network_output)[1:3] / 3, tf.int32)),
                                     max_outputs=1)

                    # beta = tf.get_default_graph().get_tensor_by_name('upsampling/deconv_5/conv5/batch_norm/beta/read:0')
                    # tf.summary.histogram('Beta', beta)

    # Evaluation
    # ----------
    if mode == tf.estimator.ModeKeys.EVAL:
        if prediction_type == PredictionType.CLASSIFICATION:
            metrics = {
                'eval/accuracy': tf.metrics.accuracy(labels, predictions=prediction_labels),
                'eval/mIOU': tf.metrics.mean_iou(labels, prediction_labels, num_classes=model_params.n_classes,)
                                                 # weights=tf.cast(training_params.weights_evaluation_miou, tf.float32))
            }
        elif prediction_type == PredictionType.REGRESSION:
            metrics = {'eval/accuracy': tf.metrics.mean_squared_error(labels, predictions=prediction_labels)}
        elif prediction_type == PredictionType.MULTILABEL:
            metrics = {'eval/MSE': tf.metrics.mean_squared_error(tf.cast(labels, tf.float32),
                                                                 predictions=prediction_probs),
                       'eval/accuracy': tf.metrics.accuracy(tf.cast(labels, tf.bool),
                                                            predictions=tf.cast(prediction_labels, tf.bool)),
                       'eval/mIOU': tf.metrics.mean_iou(labels, prediction_labels, num_classes=model_params.n_classes)
                                                        # weights=training_params.weights_evaluation_miou)
                       }
    else:
        metrics = None

    # Export
    # ------
    if mode == tf.estimator.ModeKeys.PREDICT:

        export_outputs = dict()

        if 'original_shape' in features.keys():
            with tf.name_scope('ResizeOutput'):
                resized_predictions = dict()
                # Resize all the elements in predictions
                for k, v in predictions.items():
                    # Labels is rank-3 so we need to be careful in using tf.image.resize_images
                    assert isinstance(v, tf.Tensor)
                    v2 = v if len(v.get_shape()) == 4 else v[:, :, :, None]
                    v2 = tf.image.resize_images(v2, features['original_shape'],
                                                method=tf.image.ResizeMethod.BILINEAR if v.dtype == tf.float32
                                                else tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    v2 = v2 if len(v.get_shape()) == 4 else v2[:, :, :, 0]
                    resized_predictions[k] = v2
                export_outputs['resized_output'] = tf.estimator.export.PredictOutput(resized_predictions)

            predictions['original_shape'] = features['original_shape']

        export_outputs['output'] = tf.estimator.export.PredictOutput(predictions)

        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs['output']
    else:
        export_outputs = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics,
                                      export_outputs=export_outputs,
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )
