from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np


_VGG_MEANS = [123.68, 116.78, 103.94]


def mean_substraction(input_tensor, means=_VGG_MEANS):
    return tf.subtract(input_tensor, np.array(means)[None, None, None, :], name='MeanSubstraction')


def vgg_16_fn(input_tensor: tf.Tensor, scope='vgg_16', blocks=5, weight_decay=0.0005) \
        -> (tf.Tensor, list):  # list of tf.Tensors (layers)
    intermediate_levels = []
    # intermediate_levels.append(input_tensor)
    with slim.arg_scope(nets.vgg.vgg_arg_scope(weight_decay=weight_decay)):
        with tf.variable_scope(scope, 'vgg_16', [input_tensor]) as sc:
            input_tensor = mean_substraction(input_tensor)
            intermediate_levels.append(input_tensor)
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope(
                    [layers.conv2d, layers.fully_connected, layers.max_pool2d],
                    outputs_collections=end_points_collection):
                net = layers.repeat(
                    input_tensor, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                intermediate_levels.append(net)
                net = layers.max_pool2d(net, [2, 2], scope='pool1')
                if blocks >= 2:
                    net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                    intermediate_levels.append(net)
                    net = layers.max_pool2d(net, [2, 2], scope='pool2')
                if blocks >= 3:
                    net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                    intermediate_levels.append(net)
                    net = layers.max_pool2d(net, [2, 2], scope='pool3')
                if blocks >= 4:
                    net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                    intermediate_levels.append(net)
                    net = layers.max_pool2d(net, [2, 2], scope='pool4')
                if blocks >= 5:
                    net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                    intermediate_levels.append(net)
                    net = layers.max_pool2d(net, [2, 2], scope='pool5')

                return net, intermediate_levels


def resnet_v1_50_fn(input_tensor: tf.Tensor, is_training=False, blocks=4, weight_decay=0.0001, renorm=True) -> tf.Tensor:
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=0.999)), \
         slim.arg_scope([layers.batch_norm], renorm_decay=0.95, renorm=renorm):
        input_tensor = mean_substraction(input_tensor)
        assert 0 < blocks <= 4
        blocks_list = [
              nets.resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              nets.resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              nets.resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
              nets.resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        net, endpoints = nets.resnet_v1.resnet_v1(input_tensor,
                                                  blocks=blocks_list[:blocks],
                                                  num_classes=None,
                                                  is_training=is_training,
                                                  global_pool=False,
                                                  output_stride=None,
                                                  include_root_block=True,
                                                  reuse=None,
                                                  scope='resnet_v1_50')

        desired_endpoints = ['resnet_v1_50/conv1',
                             'resnet_v1_50/block1/unit_2/bottleneck_v1',
                             'resnet_v1_50/block2/unit_3/bottleneck_v1',
                             'resnet_v1_50/block3/unit_5/bottleneck_v1',
                             'resnet_v1_50/block4/unit_2/bottleneck_v1'
                             ]

        intermediate_layers = list()
        intermediate_layers.append(input_tensor)
        for d in desired_endpoints:
            intermediate_layers.append(endpoints[d])

        return net, intermediate_layers
