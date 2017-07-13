import tensorflow as tf
from tensorflow.contrib import layers, slim


def model_fn(mode, features, labels, params):

    model_params = params['model_params']
    num_classes = params['num_classes']

    logits = inference(features['images'], model_params, num_classes,
                       is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    with tf.name_scope('softmax'):
        prediction_probs = tf.nn.softmax(logits, name='softmax')
    prediction_labels = tf.argmax(logits, axis=-1, name='label_preds')
    predictions = {'probs': prediction_probs, 'labels': prediction_labels}

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        onehot_labels = tf.one_hot(indices=labels, depth=num_classes)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels),
                                  name="loss")
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        ema = tf.train.ExponentialMovingAverage(0.9)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema.apply([loss]))
        ema_loss = ema.average(loss)
        tf.summary.scalar('losses/loss_EMA', ema_loss)
        tf.summary.scalar('losses/loss_batch', loss)
        # TODO
        #tf.summary.image('output/prediction',
        #                 tf.image.resize_images(prediction_labels, tf.cast(tf.shape(prediction_labels)[1:3]/3, tf.int32)), max_outputs=1)

        optimizer = tf.train.AdamOptimizer(params['learning_rate'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    else:
        ema_loss, train_op = None, None

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'accuracy': tf.metrics.accuracy(labels, predictions=prediction_labels)}
    else:
        metrics = None

    return tf.estimator.EstimatorSpec(mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=metrics,
                                      export_outputs={'output':
                                                      tf.estimator.export.PredictOutput({'labels': prediction_labels})}
                                      )


def inference(images, model_params, num_classes, is_training=False, weight_decay=0.0):

    """Model function for CNN.
    Args:
        :param images: Images placeholder, from inputs().
        :param filter_size: Size of the convolution kernel (One number or list of length 17)
        :param skip: Number of skip connections (Number between 0 and 3)

    Returns:
        :return softmax_linear: Output tensor with the computed logits.
    """

    def conv_pool(input_tensor, layer_params, number):
        for i, (nb_filters, filter_size) in enumerate(layer_params):
            input_tensor = layers.conv2d(
                inputs=input_tensor,
                num_outputs=nb_filters,
                kernel_size=[filter_size, filter_size],
                scope="conv{}_{}".format(number, i+1))

        pool = tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[2, 2], strides=2, name="pool{}".format(number))
        return pool

    def upsample_conv(pooled_layer, previous_layer, layer_params, number):
        with tf.name_scope('upsample{}'.format(number)):
            if previous_layer.get_shape()[1] and previous_layer.get_shape()[2]:
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
                scope="conv{}_{}".format(number, i+1)
            )
        return input_tensor

    with slim.arg_scope([layers.conv2d], activation_fn=tf.nn.relu, padding='same',
                        #normalizer_fn=layers.batch_norm,
                        weights_regularizer=slim.regularizers.l2_regularizer(weight_decay)):
        with slim.arg_scope([layers.batch_norm], is_training=is_training):
            intermediate_levels = []
            current_tensor = images
            n_layer = 1
            for layer_params in model_params:
                intermediate_levels.append(current_tensor)
                current_tensor = conv_pool(current_tensor, layer_params, n_layer)
                n_layer += 1

            for i in reversed(range(len(intermediate_levels))):
                current_tensor = upsample_conv(current_tensor, intermediate_levels[i],
                                               reversed(model_params[i]), n_layer)
                n_layer += 1

            logits = layers.conv2d(
                inputs=current_tensor,
                num_outputs=num_classes,
                activation_fn=None,
                kernel_size=[7, 7],
                scope="conv{}_logits".format(n_layer)
            )
    return logits
