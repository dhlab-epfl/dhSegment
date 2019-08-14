from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from ..model import Encoder
import os
import tarfile
from ...utils.misc import get_data_folder, download_file

_VGG_MEANS = [123.68, 116.78, 103.94]


def mean_substraction(input_tensor, means=_VGG_MEANS):
    return tf.subtract(input_tensor, np.array(means)[None, None, None, :], name='MeanSubstraction')


class VGG16(Encoder):
    """VGG-16 implementation

    :ivar blocks: number of blocks (vgg blocks)
    :ivar weight_decay: weight decay value
    :ivar pretrained_file: path to the file (.ckpt) containing the pretrained weights
    """
    def __init__(self, blocks: int=5, weight_decay: float=0.0005):
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

    def __call__(self, images: tf.Tensor, is_training=False):
        outputs = []

        with slim.arg_scope(nets.vgg.vgg_arg_scope(weight_decay=self.weight_decay)):
            with tf.variable_scope(None, 'vgg_16', [images]) as sc:
                input_tensor = mean_substraction(images)
                outputs.append(input_tensor)
                end_points_collection = sc.original_name_scope + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope(
                        [layers.conv2d, layers.fully_connected, layers.max_pool2d],
                        outputs_collections=end_points_collection):
                    net = layers.repeat(
                        input_tensor, 2, layers.conv2d, 64, [3, 3], scope='conv1')
                    net = layers.max_pool2d(net, [2, 2], scope='pool1')
                    outputs.append(net)
                    if self.blocks >= 2:
                        net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
                        net = layers.max_pool2d(net, [2, 2], scope='pool2')
                        outputs.append(net)
                    if self.blocks >= 3:
                        net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
                        net = layers.max_pool2d(net, [2, 2], scope='pool3')
                        outputs.append(net)
                    if self.blocks >= 4:
                        net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
                        net = layers.max_pool2d(net, [2, 2], scope='pool4')
                        outputs.append(net)
                    if self.blocks >= 5:
                        net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
                        net = layers.max_pool2d(net, [2, 2], scope='pool5')
                        outputs.append(net)

                    return outputs
