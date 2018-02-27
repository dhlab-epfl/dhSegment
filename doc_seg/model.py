import tensorflow as tf
from tensorflow.contrib import layers  # TODO migration to tf.layers ?
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim import arg_scope
from .utils import PredictionType, class_to_label_image, ModelParams, TrainingParams
from .pretrained_models import vgg_16_fn, resnet_v1_50_fn
from doc_seg import utils
import numpy as np


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
    else:
        raise NotImplementedError

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Pretrained weights as initialization
        pretrained_restorer = tf.train.Saver(var_list=[v for v in tf.global_variables()
                                                       if key_restore_model in v.name])

        def init_fn(scaffold, session):
            pretrained_restorer.restore(session, model_params.pretrained_model_file)
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
            prediction_labels = tf.greater_equal(prediction_probs, 0.5, name='labels')  # [B,H,W,C]
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
            labels_visualization = utils.multiclass_to_label_image(labels_visualization, classes_file)
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

    # Evaluation
    # ----------
    if mode == tf.estimator.ModeKeys.EVAL:
        if prediction_type == PredictionType.CLASSIFICATION:
            metrics = {'accuracy': tf.metrics.accuracy(labels, predictions=prediction_labels)}
        elif prediction_type == PredictionType.REGRESSION:
            metrics = {'accuracy': tf.metrics.mean_squared_error(labels, predictions=prediction_labels)}
        elif prediction_type == PredictionType.MULTILABEL:
            metrics = {'eval/MSE': tf.metrics.mean_squared_error(tf.cast(labels, tf.float32),
                                                                 predictions=prediction_probs),
                       'eval/accuracy': tf.metrics.accuracy(tf.cast(labels, tf.bool),
                                                            predictions=tf.cast(prediction_labels, tf.bool))
                       }
    else:
        metrics = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics,
                                      export_outputs={'output':
                                                          tf.estimator.export.PredictOutput(predictions)},
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )


def inference_vgg16(images: tf.Tensor, params: ModelParams, num_classes: int, use_batch_norm=False, weight_decay=0.0,
                    is_training=False) -> tf.Tensor:
    with tf.name_scope('vgg_augmented'):

        if use_batch_norm:
            if params.batch_renorm:
                renorm_clipping = {'rmax': 100, 'rmin': 0.1, 'dmax': 10}
                renorm_momentum = 0.98
            else:
                renorm_clipping = None
                renorm_momentum = 0.99
            batch_norm_fn = lambda x: tf.layers.batch_normalization(x, axis=-1, training=is_training, name='batch_norm',
                                                                    renorm=params.batch_renorm,
                                                                    renorm_clipping=renorm_clipping,
                                                                    renorm_momentum=renorm_momentum)
        else:
            batch_norm_fn = None

        def upsample_conv(pooled_layer, previous_layer, layer_params, number):
            with tf.name_scope('deconv{}'.format(number)):
                if previous_layer.get_shape()[1].value and previous_layer.get_shape()[2].value:
                    target_shape = previous_layer.get_shape()[1:3]
                else:
                    target_shape = tf.shape(previous_layer)[1:3]
                upsampled_layer = tf.image.resize_images(pooled_layer, target_shape,
                                                         method=tf.image.ResizeMethod.BILINEAR)
                input_tensor = tf.concat([upsampled_layer, previous_layer], 3)

                for i, (nb_filters, filter_size) in enumerate(layer_params):
                    input_tensor = layers.conv2d(
                        inputs=input_tensor,
                        num_outputs=nb_filters,
                        kernel_size=[filter_size, filter_size],
                        normalizer_fn=batch_norm_fn,
                        scope="conv{}_{}".format(number, i + 1)
                    )
            return input_tensor

        # Original VGG :
        vgg_net, intermediate_levels = vgg_16_fn(images, blocks=5, weight_decay=weight_decay)
        out_tensor = vgg_net

        # Intermediate convolution
        if params.intermediate_conv is not None:
            with tf.name_scope('intermediate_convs'):
                for layer_params in params.intermediate_conv:
                    for k, (nb_filters, filter_size) in enumerate(layer_params):
                        out_tensor = layers.conv2d(inputs=out_tensor,
                                                   num_outputs=nb_filters,
                                                   kernel_size=[filter_size, filter_size],
                                                   normalizer_fn=batch_norm_fn,
                                                   scope='conv_{}'.format(k + 1))

        # Upsampling :
        with tf.name_scope('upsampling'):
            selected_upscale_params = [l for i, l in enumerate(params.upscale_params)
                                       if params.selected_levels_upscaling[i]]

            assert len(params.selected_levels_upscaling) == len(intermediate_levels), \
                'Upscaling : {} is different from {}'.format(len(params.selected_levels_upscaling),
                                                             len(intermediate_levels))

            selected_intermediate_levels = [l for i, l in enumerate(intermediate_levels)
                                            if params.selected_levels_upscaling[i]]

            # Upsampling loop
            n_layer = 1
            for i in reversed(range(len(selected_intermediate_levels))):
                out_tensor = upsample_conv(out_tensor, selected_intermediate_levels[i],
                                           selected_upscale_params[i], n_layer)
                n_layer += 1

            logits = layers.conv2d(inputs=out_tensor,
                                   num_outputs=num_classes,
                                   activation_fn=None,
                                   kernel_size=[1, 1],
                                   scope="conv{}-logits".format(n_layer))

        return logits  # [B,h,w,Classes]


def inference_resnet_v1_50(images, params, num_classes, use_batch_norm=False, weight_decay=0.0,
                           is_training=False) -> tf.Tensor:
    with tf.name_scope('resnet_augmented'):
        if use_batch_norm:
            if params.batch_renorm:
                renorm_clipping = {'rmax': 100, 'rmin': 0.1, 'dmax': 1}
                renorm_momentum = 0.98
            else:
                renorm_clipping = None
                renorm_momentum = 0.99
            batch_norm_fn = lambda x: tf.layers.batch_normalization(x, axis=-1, training=is_training, name='batch_norm',
                                                                    renorm=params.batch_renorm,
                                                                    renorm_clipping=renorm_clipping,
                                                                    renorm_momentum=renorm_momentum)
        else:
            batch_norm_fn = None

        def upsample_conv(input_tensor, previous_intermediate_layer, layer_params, number):
            with tf.name_scope('deconv_{}'.format(number)):
                if previous_intermediate_layer.get_shape()[1].value and \
                        previous_intermediate_layer.get_shape()[2].value:
                    target_shape = previous_intermediate_layer.get_shape()[1:3]
                else:
                    target_shape = tf.shape(previous_intermediate_layer)[1:3]
                upsampled_layer = tf.image.resize_images(input_tensor, target_shape,
                                                         method=tf.image.ResizeMethod.BILINEAR)
                net = tf.concat([upsampled_layer, previous_intermediate_layer], 3)

                filter_size, nb_bottlenecks = layer_params
                if nb_bottlenecks > 0:
                    for i in range(nb_bottlenecks):
                        net = resnet_v1.bottleneck(
                            inputs=net,
                            depth=filter_size,
                            depth_bottleneck=filter_size // 4,
                            stride=1
                        )
                else:
                    net = layers.conv2d(
                            inputs=net,
                            num_outputs=filter_size,
                            kernel_size=[1, 1],
                            scope="conv{}".format(number)
                        )

            return net

        # Original ResNet
        blocks_needed = max([i - 1 for i, is_needed in enumerate(params.selected_levels_upscaling) if is_needed])
        resnet_net, intermediate_layers = resnet_v1_50_fn(images, is_training=False, blocks=blocks_needed,
                                                          weight_decay=weight_decay, renorm=False)

        # Upsampling
        with tf.name_scope('upsampling'):
            with arg_scope([layers.conv2d],
                           normalizer_fn=batch_norm_fn,
                           weights_regularizer=layers.l2_regularizer(weight_decay)):
                selected_upscale_params = [l for i, l in enumerate(params.upscale_params)
                                           if params.selected_levels_upscaling[i]]

                assert len(params.selected_levels_upscaling) == len(intermediate_layers), \
                    'Upscaling : {} is different from {}'.format(len(params.selected_levels_upscaling),
                                                                 len(intermediate_layers))

                selected_intermediate_levels = [l for i, l in enumerate(intermediate_layers)
                                                if params.selected_levels_upscaling[i]]

                selected_intermediate_levels.insert(images, 0)

                # Force layers to not be too big to reduce memory usage
                for i, l in enumerate(selected_intermediate_levels):
                    if l.get_shape()[-1] > params.max_depth:
                        selected_intermediate_levels[i] = layers.conv2d(
                            inputs=l,
                            num_outputs=params.max_depth,
                            kernel_size=[1, 1],
                            scope="dimreduc_{}".format(i),
                            activation_fn=None
                        )

                # Deconvolving loop
                out_tensor = selected_intermediate_levels[-1]
                n_layer = 1
                for i in reversed(range(len(selected_intermediate_levels)))[1:]:
                    out_tensor = upsample_conv(out_tensor, selected_intermediate_levels[i],
                                               selected_upscale_params[i], n_layer)

                    n_layer += 1

                if images.get_shape()[1].value and images.get_shape()[2].value:
                    target_shape = images.get_shape()[1:3]
                else:
                    target_shape = tf.shape(images)[1:3]
                out_tensor = tf.image.resize_images(out_tensor, target_shape,
                                                    method=tf.image.ResizeMethod.BILINEAR)

            logits = layers.conv2d(inputs=out_tensor,
                                   num_outputs=num_classes,
                                   activation_fn=None,
                                   kernel_size=[1, 1],
                                   scope="conv{}-logits".format(n_layer))

        return logits
