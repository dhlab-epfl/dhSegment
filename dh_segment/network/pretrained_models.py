from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from .model import Encoder
import os
import tarfile
from ..utils.misc import get_data_folder, download_file

_VGG_MEANS = [123.68, 116.78, 103.94]


def mean_substraction(input_tensor, means=_VGG_MEANS):
    return tf.subtract(input_tensor, np.array(means)[None, None, None, :], name='MeanSubstraction')


class ResnetV1_50(Encoder):
    def __init__(self, train_batchnorm=False, blocks=4, weight_decay=0.0001,
                    renorm=True, corrected_version=False):
        self.train_batchnorm = train_batchnorm
        self.blocks = blocks
        self.weight_decay = weight_decay
        self.renorm = renorm
        self.corrected_version = corrected_version
        self.pretrained_file = os.path.join(get_data_folder(), 'resnet_v1_50.ckpt')
        if not os.path.exists(self.pretrained_file):
            print("Could not find pre-trained file {}, downloading it!".format(self.pretrained_file))
            tar_filename = os.path.join(get_data_folder(), 'resnet_v1_50.tar.gz')
            download_file('http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz', tar_filename)
            tar = tarfile.open(tar_filename)
            tar.extractall(path=get_data_folder())
            tar.close()
            os.remove(tar_filename)
            assert os.path.exists(self.pretrained_file)
            print('Pre-trained weights downloaded!')

    def pretrained_information(self):
        return self.pretrained_file, [v for v in tf.global_variables()
                                      if 'resnet_v1_50' in v.name
                                      and 'renorm' not in v.name]

    def __call__(self, images: tf.Tensor):
        outputs = []

        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope(weight_decay=self.weight_decay, batch_norm_decay=0.999)), \
             slim.arg_scope([layers.batch_norm], renorm_decay=0.95, renorm=self.renorm):
            mean_substracted_tensor = mean_substraction(images)
            assert 0 < self.blocks <= 4

            if self.corrected_version:
                def corrected_resnet_v1_block(scope, base_depth, num_units, stride):
                    """Helper function for creating a resnet_v1 bottleneck block.

                    Args:
                      scope: The scope of the block.
                      base_depth: The depth of the bottleneck layer for each unit.
                      num_units: The number of units in the block.
                      stride: The stride of the block, implemented as a stride in the last unit.
                        All other units have stride=1.

                    Returns:
                      A resnet_v1 bottleneck block.
                    """
                    return nets.resnet_utils.Block(scope, nets.resnet_v1.bottleneck, [{
                        'depth': base_depth * 4,
                        'depth_bottleneck': base_depth,
                        'stride': stride
                    }] + [{
                        'depth': base_depth * 4,
                        'depth_bottleneck': base_depth,
                        'stride': 1
                    }] * (num_units - 1))

                blocks_list = [
                    corrected_resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
                    corrected_resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                    corrected_resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
                    corrected_resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
                ]
                desired_endpoints = [
                    'resnet_v1_50/conv1',
                    'resnet_v1_50/block1/unit_3/bottleneck_v1',
                    'resnet_v1_50/block2/unit_4/bottleneck_v1',
                    'resnet_v1_50/block3/unit_6/bottleneck_v1',
                    'resnet_v1_50/block4/unit_3/bottleneck_v1'
                ]
            else:
                blocks_list = [
                    nets.resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                    nets.resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                    nets.resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
                    nets.resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                ]
                desired_endpoints = [
                    'resnet_v1_50/conv1',
                    'resnet_v1_50/block1/unit_2/bottleneck_v1',
                    'resnet_v1_50/block2/unit_3/bottleneck_v1',
                    'resnet_v1_50/block3/unit_5/bottleneck_v1',
                    'resnet_v1_50/block4/unit_3/bottleneck_v1'
                ]

            net, endpoints = nets.resnet_v1.resnet_v1(mean_substracted_tensor,
                                                      blocks=blocks_list[:self.blocks],
                                                      num_classes=None,
                                                      is_training=self.train_batchnorm,
                                                      global_pool=False,
                                                      output_stride=None,
                                                      include_root_block=True,
                                                      reuse=None,
                                                      scope='resnet_v1_50')

            # Add standardized original images
            outputs.append(mean_substracted_tensor/127.0)

            for d in desired_endpoints[:self.blocks + 1]:
                outputs.append(endpoints[d])

            return outputs


class VGG16(Encoder):
    def __init__(self, blocks=5, weight_decay=0.0005):
        self.blocks = blocks
        self.weight_decay = weight_decay
        self.pretrained_file = os.path.join(get_data_folder(), 'vgg_16.ckpt')
        if not os.path.exists(self.pretrained_file):
            print("Could not find pre-trained file {}, downloading it!".format(self.pretrained_file))
            tar_filename = os.path.join(get_data_folder(), 'vgg_16.tar.gz')
            download_file('http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz', tar_filename)
            tar = tarfile.open(tar_filename)
            tar.extractall(path=get_data_folder())
            tar.close()
            os.remove(tar_filename)
            assert os.path.exists(self.pretrained_file)
            print('Pre-trained weights downloaded!')

    def pretrained_information(self):
        return self.pretrained_file, [v for v in tf.global_variables()
                                      if 'vgg_16' in v.name
                                      and 'renorm' not in v.name]

    def __call__(self, images: tf.Tensor):
        intermediate_levels = []

        with slim.arg_scope(nets.vgg.vgg_arg_scope(weight_decay=self.weight_decay)):
            with tf.variable_scope(None, 'vgg_16', [images]) as sc:
                input_tensor = mean_substraction(images)
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
                    if self.blocks >= 2:
                        net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                        intermediate_levels.append(net)
                        net = layers.max_pool2d(net, [2, 2], scope='pool2')
                    if self.blocks >= 3:
                        net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                        intermediate_levels.append(net)
                        net = layers.max_pool2d(net, [2, 2], scope='pool3')
                    if self.blocks >= 4:
                        net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                        intermediate_levels.append(net)
                        net = layers.max_pool2d(net, [2, 2], scope='pool4')
                    if self.blocks >= 5:
                        net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                        intermediate_levels.append(net)
                        net = layers.max_pool2d(net, [2, 2], scope='pool5')

                    return intermediate_levels
