import tensorflow as tf
from tensorflow.contrib import layers, slim  # TODO migration to tf.layers ?
from .utils import PredictionType, class_to_label_image
from .pretrained_models import vgg_16_fn, resnet_v1_50_fn


def model_fn(mode, features, labels, params):

    model_params = params['vgg_conv_params']
    num_classes = params['num_classes']
    prediction_type = params['prediction_type']

    network_output = inference_vgg16(features['images'],
                                     model_params,
                                     num_classes,
                                     weight_decay=params['weight_decay'],
                                     )

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Pretrained weights as initialization
        pretrained_restorer = tf.train.Saver(var_list=[v for v in tf.global_variables()
                                                       if 'vgg_16' in v.name])

        def init_fn(scaffold, session):
            pretrained_restorer.restore(session, params['pretrained_file'])
    else:
        init_fn = None

    # network_output = inference(features['images'], model_params, num_classes,
    #                            is_training=(mode == tf.estimator.ModeKeys.TRAIN),
    #                            use_batch_norm=params['batch_norm'],
    #                            weight_decay=params['weight_decay'])

    # Upscale the output here -> TODO try upscaling afterwards ?
    with tf.name_scope('Upsampling'):
        output_shape = tf.shape(network_output)[1:3]
        network_output = tf.image.resize_images(network_output, output_shape*32, method=tf.image.ResizeMethod.BILINEAR)

    # Prediction
    if prediction_type == PredictionType.CLASSIFICATION:
        with tf.name_scope('softmax'):
            prediction_probs = tf.nn.softmax(network_output, name='softmax')
        prediction_labels = tf.argmax(network_output, axis=-1, name='label_preds')
        predictions = {'probs': prediction_probs, 'labels': prediction_labels}
    elif prediction_type == PredictionType.REGRESSION:
        predictions = {'output_values': network_output}
    else:
        raise NotImplementedError

    # Loss
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        regularized_loss = tf.losses.get_regularization_loss()
        if prediction_type == PredictionType.CLASSIFICATION:
            onehot_labels = tf.one_hot(indices=labels, depth=num_classes)
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network_output,
                                                                              labels=onehot_labels),
                                      name="loss")
        elif prediction_type == PredictionType.REGRESSION:
            loss = tf.losses.mean_squared_error(labels, network_output)

        loss += regularized_loss
    else:
        loss = None

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        ema = tf.train.ExponentialMovingAverage(0.9)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        ema_loss = ema.average(loss)
        tf.summary.scalar('losses/loss', ema_loss)
        tf.summary.scalar('losses/loss_per_batch', loss)
        tf.summary.scalar('losses/regularized_loss', regularized_loss)
        if prediction_type == PredictionType.CLASSIFICATION:
            tf.summary.image('output/prediction',
                             tf.image.resize_images(class_to_label_image(prediction_labels, params['classes_file']),
                                                    tf.cast(tf.shape(network_output)[1:3]/3, tf.int32)), max_outputs=1)
            tf.summary.image('output/probs',
                             tf.image.resize_images(prediction_probs[:, :, :, 1:],
                                                    tf.cast(tf.shape(network_output)[1:3] / 3, tf.int32)), max_outputs=1)
        elif prediction_type == PredictionType.REGRESSION:
            tf.summary.image('output/prediction', network_output[:, :, :, 0:1], max_outputs=1)

        if params['exponential_learning']:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(params['learning_rate'], global_step, 200, 0.95, staircase=False)
        else:
            learning_rate = params['learning_rate']
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        ema_loss, train_op = None, None

    if prediction_type == PredictionType.CLASSIFICATION:
        prediction_labels = prediction_labels
    elif prediction_type == PredictionType.REGRESSION:
        prediction_labels = network_output
    else:
        raise NotImplementedError

    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        if prediction_type == PredictionType.CLASSIFICATION:
            metrics = {'accuracy': tf.metrics.accuracy(labels, predictions=prediction_labels)}
        elif prediction_type == PredictionType.REGRESSION:
            metrics = {'accuracy': tf.metrics.mean_squared_error(labels, predictions=prediction_labels)}
        else:
            raise NotImplementedError
    else:
        metrics = None

    # return tf.estimator.EstimatorSpec(mode,
    #                                   predictions=predictions,
    #                                   loss=loss,
    #                                   train_op=train_op,
    #                                   eval_metric_ops=metrics,
    #                                   export_outputs={'output':
    #                                                   tf.estimator.export.PredictOutput(predictions)}
    #                                   )

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics,
                                      export_outputs={'output':
                                                      tf.estimator.export.PredictOutput(predictions)},
                                      scaffold=tf.train.Scaffold(init_fn=init_fn)
                                      )


def inference(images, all_layer_params, num_classes, is_training=False, use_batch_norm=False, weight_decay=0.0):
    """

    :param images: images tensor
    :param all_layer_params: List of List of Tuple(nb_filters, filter_size)
    :param num_classes: Dimension of the output for each pixel
    :param is_training:
    :param use_batch_norm:
    :param weight_decay:
    :return: Linear activations
    """

    if use_batch_norm:
        # TODO use renorm
        batch_norm_fn = lambda x: tf.layers.batch_normalization(x, axis=-1, training=is_training, name='batch_norm')
    else:
        batch_norm_fn = None

    def conv_pool(input_tensor, layer_params, number):
        for i, (nb_filters, filter_size) in enumerate(layer_params):
            input_tensor = layers.conv2d(
                inputs=input_tensor,
                num_outputs=nb_filters,
                kernel_size=[filter_size, filter_size],
                normalizer_fn=batch_norm_fn,
                scope="conv{}_{}".format(number, i+1))

        pool = tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[2, 2], strides=2, name="pool{}".format(number))
        return pool

    def upsample_conv(pooled_layer, previous_layer, layer_params, number):
        with tf.name_scope('upsample{}'.format(number)):
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
                scope="conv{}_{}".format(number, i+1)
            )
        return input_tensor

    with slim.arg_scope([layers.conv2d], activation_fn=tf.nn.relu, padding='same',
                        normalizer_fn=layers.batch_norm if use_batch_norm else None,
                        weights_regularizer=slim.regularizers.l2_regularizer(weight_decay)):
        with slim.arg_scope([layers.batch_norm], is_training=is_training):
            intermediate_levels = []
            current_tensor = images
            n_layer = 1
            for layer_params in all_layer_params:
                intermediate_levels.append(current_tensor)
                current_tensor = conv_pool(current_tensor, layer_params, n_layer)
                n_layer += 1

            for i in reversed(range(len(intermediate_levels))):
                current_tensor = upsample_conv(current_tensor, intermediate_levels[i],
                                               reversed(all_layer_params[i]), n_layer)
                n_layer += 1

            logits = layers.conv2d(
                inputs=current_tensor,
                num_outputs=num_classes,
                activation_fn=None,
                kernel_size=[7, 7],
                scope="conv{}_logits".format(n_layer)
            )
    return logits


def inference_vgg16(images, model_params, num_classes, use_batch_norm=False, weight_decay=0.0) -> tf.Tensor:
    with tf.name_scope('vgg_augmented'):
        # If we have several conv layers
        vgg_conv_params = model_params

        vgg_net = vgg_16_fn(images, blocks=5, weight_decay=weight_decay)
        out_tensor = vgg_net

        for i, (nb_filters, filter_size) in enumerate(vgg_conv_params):
            out_tensor = layers.conv2d(inputs=out_tensor,
                                       num_outputs=nb_filters,
                                       kernel_size=[filter_size, filter_size],
                                       scope="conv6-{}".format(i + 1)
                                       )

        logits = layers.conv2d(inputs=out_tensor,
                               num_outputs=num_classes,
                               activation_fn=None,
                               kernel_size=[1, 1],
                               scope="conv6-logits")

        return logits  # [B,h,w,Classes]


def inference_resnet_v1_50(images, model_params, num_classes, use_batch_norm=False, weight_decay=0.0) -> tf.Tensor:
    with tf.name_scope('resnet_v1_50'):
        resnet_conv_params = model_params

        resnet_net = resnet_v1_50_fn(images, is_training=False, blocks=4, weight_decay=weight_decay)
        out_tensor = resnet_net

        for i, (nb_filters, filter_size) in enumerate(resnet_conv_params):
            out_tensor = layers.conv2d(inputs=out_tensor,
                                       num_outputs=nb_filters,
                                       kernel_size=[filter_size, filter_size],
                                       scope="conv-{}".format(i + 1)
                                       )

        logits = layers.conv2d(inputs=out_tensor,
                               num_outputs=num_classes,
                               activation_fn=None,
                               kernel_size=[1, 1],
                               scope="conv-logits")

        return logits