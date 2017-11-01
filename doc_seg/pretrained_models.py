from tensorflow.contrib import slim, layers
import tensorflow as tf
from tensorflow.contrib.slim import nets
import numpy as np
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import bottleneck, resnet_v1_block
from tensorflow.python.ops import math_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope


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


@add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
    # Original resnet_utils.stack_blocks_dense modified to also output intermediate layers
    """Stacks ResNet `Blocks` and controls output feature density.

    First, this function creates scopes for the ResNet in the form of
    'block_name/unit_1', 'block_name/unit_2', etc.

    Second, this function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.

    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This results
    to a nominal ResNet output_stride equal to 8. If we set the output_stride to
    half the nominal network stride (e.g., output_stride=4), then we compute
    responses twice.

    Control of the output feature density is implemented by atrous convolution.

    Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

    Returns:
    net: Output tensor with stride equal to the specified output_stride.

    Raises:
    ValueError: If the target output_stride is not valid.
    """
    intermediate_layers = list()
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    #  atrous convolution with stride=1 and multiply the atrous rate by the
                    #  current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)
        intermediate_layers.append(net)  # add layer after each block

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net, intermediate_layers


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=None,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
    """Generator for v1 ResNet models.
    This function generates a family of ResNet v1 models. See the resnet_v1_*()
    methods for specific model instantiations, obtained by selecting different
    block instantiations that produce ResNets of various depths.
    Training for image classification on Imagenet is usually done with [224, 224]
    inputs, resulting in [7, 7] feature maps at the output of the last ResNet
    block for the ResNets defined in [1] that have nominal stride equal to 32.
    However, for dense prediction tasks we advise that one uses inputs with
    spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
    this case the feature maps at the ResNet output will have spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
    and corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input [225, 225]
    images results in [8, 8] feature maps at the output of the last ResNet block.
    For dense prediction tasks, the ResNet needs to run in fully-convolutional
    (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
    have nominal stride equal to 32 and a good choice in FCN mode is to use
    output_stride=16 in order to increase the density of the computed features at
    small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      blocks: A list of length equal to the number of ResNet blocks. Each element
        is a resnet_utils.Block object describing the units in the block.
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not. If None, the value inherited from
        the resnet_arg_scope is used. Specifying value None is deprecated.
      global_pool: If True, we perform global average pooling before computing the
        logits. Set to True for image classification, False for dense prediction.
      output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
      include_root_block: If True, include the initial convolution followed by
        max-pooling, if False excludes it.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is False, then height_out and width_out are reduced by a
        factor of output_stride compared to the respective height_in and width_in,
        else both height_out and width_out equal one. If num_classes is None, then
        net is the output of the last ResNet block, potentially after global
        average pooling. If num_classes is not None, net contains the pre-softmax
        activations.
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    intermediate_layers = list()
    with tf.variable_scope(
            scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with arg_scope(
                [layers.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
                outputs_collections=end_points_collection):
            if is_training is not None:
                bn_scope = arg_scope([layers.batch_norm], is_training=is_training)
            else:
                bn_scope = arg_scope([])
            with bn_scope:
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    intermediate_layers.append(net)
                    net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net, inter_layers = stack_blocks_dense(net, blocks, output_stride)
                intermediate_layers += inter_layers
                if global_pool:
                    # Global average pooling.
                    net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = layers.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='logits')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = layers_lib.softmax(
                        net, scope='predictions')
                return net, intermediate_layers, end_points


def resnet_v1_50_fn(input_tensor: tf.Tensor,
                    is_training=False,
                    blocks=4,
                    weight_decay=0.0001,
                    renorm=True) -> tf.Tensor:
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
        net, intermediate_layers = resnet_v1(input_tensor,
                                             blocks=blocks_list[:blocks],
                                             num_classes=None,
                                             is_training=is_training,
                                             global_pool=False,
                                             output_stride=None,
                                             include_root_block=True,
                                             reuse=None,
                                             scope='resnet_v1_50')[:2]

        return net, intermediate_layers
