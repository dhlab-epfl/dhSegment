#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib import layers  # TODO migration to tf.layers ?
from tensorflow.contrib.slim import arg_scope
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional, Dict


class Encoder(ABC):
    @abstractmethod
    def __call__(self, images: tf.Tensor) -> List[tf.Tensor]:
        """

        :param images: [NxHxWx3] float32 [0..255] input images
        :return: a list of the feature maps in decreasing spatial resolution (first element is most likely the input \
        image itself, then the output of the first pooling op, etc...)
        """
        pass

    def pretrained_information(self) -> Tuple[Optional[str], Union[None, List, Dict]]:
        """

        :return: The filename of the pretrained checkpoint and the corresponding variables (List of Dict mapping) \
        or `None` if no-pretraining is done
        """
        return None, None


class Decoder(ABC):
    @abstractmethod
    def __call__(self, feature_maps: List[tf.Tensor], num_classes: int) -> tf.Tensor:
        """

        :param feature_maps: list of feature maps, in decreasing spatial resolution, first one being at the original \
        resolution
        :return: [N,H,W,num_classes] float32 tensor of logit scores
        """
        pass


class SimpleDecoder(Decoder):
    """

    :ivar upsampling_dims:
    :ivar max_depth:
    :ivar weight_decay:
    :ivar self.batch_norm_fn:
    """
    def __init__(self, upsampling_dims: List[int], max_depth: int = None, train_batchnorm: bool=False,
                 weight_decay: float=0.):
        self.upsampling_dims = upsampling_dims
        self.max_depth = max_depth
        self.weight_decay = weight_decay
        if train_batchnorm:
            # TODO
            renorm = False
            if renorm:
                renorm_clipping = {'rmax': 100, 'rmin': 0.1, 'dmax': 10}
                renorm_momentum = 0.98
            else:
                renorm_clipping = None
                renorm_momentum = 0.99
            self.batch_norm_fn = lambda x: tf.layers.batch_normalization(x, axis=-1, training=train_batchnorm,
                                                                         name='batch_norm',
                                                                         renorm=renorm,
                                                                         renorm_clipping=renorm_clipping,
                                                                         renorm_momentum=renorm_momentum)
        else:
            self.batch_norm_fn = None

    def __call__(self, feature_maps: List[tf.Tensor], num_classes: int):

        # Upsampling
        with tf.variable_scope('SimpleDecoder'):
            with arg_scope([layers.conv2d],
                           normalizer_fn=self.batch_norm_fn,
                           weights_regularizer=layers.l2_regularizer(self.weight_decay)):

                assert len(self.upsampling_dims) + 1 == len(feature_maps), \
                    'Upscaling : length of {} does not match {}'.format(len(self.upsampling_dims),
                                                                        len(feature_maps))

                # Force layers to not be too big to reduce memory usage
                for i, l in enumerate(feature_maps):
                    if self.max_depth and l.get_shape()[-1] > self.max_depth:
                        feature_maps[i] = layers.conv2d(
                            inputs=l,
                            num_outputs=self.max_depth,
                            kernel_size=[1, 1],
                            scope="dimreduc_{}".format(i),
                            normalizer_fn=self.batch_norm_fn,
                            activation_fn=None
                        )

                # Deconvolving loop
                out_tensor = feature_maps[-1]
                for i, f_map in reversed(list(enumerate(feature_maps[:-1]))):
                    out_tensor = _upsample_concat(out_tensor, f_map, scope_name='upsample_{}'.format(i))
                    out_tensor = layers.conv2d(inputs=out_tensor,
                                               num_outputs=self.upsampling_dims[i],
                                               kernel_size=[3, 3],
                                               scope="conv_{}".format(i))

            logits = layers.conv2d(inputs=out_tensor,
                                   num_outputs=num_classes,
                                   activation_fn=None,
                                   kernel_size=[1, 1],
                                   scope="conv-logits")

        return logits


def _get_image_shape_tensor(tensor: tf.Tensor) -> Union[Tuple[int, int], tf.Tensor]:
    """
    Get the image shape of the tensor

    :param tensor: Input image tensor [N,H,W,...]
    :return: a (int, int) tuple if shape is defined, otherwise the corresponding tf.Tensor value
    """
    if tensor.get_shape()[1].value and \
            tensor.get_shape()[2].value:
        target_shape = tensor.get_shape()[1:3]
    else:
        target_shape = tf.shape(tensor)[1:3]
    return target_shape


def _upsample_concat(pooled_layer: tf.Tensor, previous_layer: tf.Tensor, scope_name: str='UpsampleConcat'):
    """

    :param pooled_layer: [N,H,W,C] coarse layer
    :param previous_layer: [N,H',W',C'] fine layer (H'>H, and W'>W)
    :param scope_name:
    :return: [N,H',W',C+C'] concatenation of upsampled-`pooled_layer` and `previous_layer`
    """
    with tf.name_scope(scope_name):
        # Upsamples the coarse level
        target_shape = _get_image_shape_tensor(previous_layer)
        upsampled_layer = tf.image.resize_images(pooled_layer, target_shape,
                                                 method=tf.image.ResizeMethod.BILINEAR)
        # Concatenate the upsampled-coarse and the other feature_map
        input_tensor = tf.concat([upsampled_layer, previous_layer], 3)
    return input_tensor
