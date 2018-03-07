#!/usr/bin/env python

import tensorflow as tf
from .utils import ModelParams
from tensorflow.contrib import layers  # TODO migration to tf.layers ?
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim import arg_scope
from .pretrained_models import vgg_16_fn, resnet_v1_50_fn


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
        """
        Deconvolution (upscaling) layers
        :param input_tensor:
        :param previous_intermediate_layer:
        :param layer_params:
        :param number:
        :return:
        """
        with tf.variable_scope('deconv_{}'.format(number)):
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
                        kernel_size=[3, 3],
                        scope="conv{}".format(number)
                    )

        return net

    # Original ResNet
    blocks_needed = max([i for i, is_needed in enumerate(params.selected_levels_upscaling) if is_needed])
    resnet_net, intermediate_layers = resnet_v1_50_fn(images, is_training=False, blocks=blocks_needed,
                                                      weight_decay=weight_decay, renorm=False,
                                                      corrected_version=params.correct_resnet_version)

    # Upsampling
    with tf.variable_scope('upsampling'):
        with arg_scope([layers.conv2d],
                       normalizer_fn=batch_norm_fn,
                       weights_regularizer=layers.l2_regularizer(weight_decay)):
            selected_upscale_params = [l for i, l in enumerate(params.upscale_params)
                                       if params.selected_levels_upscaling[i]]

            assert len(selected_upscale_params) == len(intermediate_layers), \
                'Upscaling : {} is different from {}'.format(len(selected_upscale_params),
                                                             len(intermediate_layers))

            selected_intermediate_levels = [l for i, l in enumerate(intermediate_layers)
                                            if params.selected_levels_upscaling[i]]

            selected_intermediate_levels.insert(0, images)

            # Force layers to not be too big to reduce memory usage
            for i, l in enumerate(selected_intermediate_levels):
                if l.get_shape()[-1] > params.max_depth:
                    selected_intermediate_levels[i] = layers.conv2d(
                        inputs=l,
                        num_outputs=params.max_depth,
                        kernel_size=[1, 1],
                        scope="dimreduc_{}".format(i),
                        # normalizer_fn=batch_norm_fn,
                        activation_fn=None
                    )

            # Deconvolving loop
            out_tensor = selected_intermediate_levels[-1]
            n_layer = 1
            for i in reversed(range(len(selected_intermediate_levels) - 1)):
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
