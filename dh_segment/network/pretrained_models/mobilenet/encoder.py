from ...model import Encoder
import tensorflow as tf
from .mobilenet_v2 import training_scope, mobilenet_base
from typing import Tuple, Optional, Union, List, Dict
from tensorflow.contrib import slim
import os
from ....utils.misc import get_data_folder, download_file
import tarfile


class MobileNetV2(Encoder):
    def __init__(self, train_batchnorm: bool=False, weight_decay: float=0.00004, batch_renorm: bool=True):
        self.train_batchnorm = train_batchnorm
        self.weight_decay = weight_decay
        self.batch_renorm = batch_renorm
        pretrained_dir = os.path.join(get_data_folder(), 'mobilenet_v2')
        self.pretrained_file = os.path.join(pretrained_dir, 'mobilenet_v2_1.0_224.ckpt')
        if not os.path.exists(self.pretrained_file+'.index'):
            print("Could not find pre-trained file {}, downloading it!".format(self.pretrained_file))
            tar_filename = os.path.join(get_data_folder(), 'resnet_v1_50.tar.gz')
            download_file('https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz', tar_filename)
            tar = tarfile.open(tar_filename)
            tar.extractall(path=pretrained_dir)
            tar.close()
            os.remove(tar_filename)
            assert os.path.exists(self.pretrained_file+'.index')
            print('Pre-trained weights downloaded!')

    def __call__(self, images: tf.Tensor, is_training=False) -> List[tf.Tensor]:
        outputs = []

        with slim.arg_scope(training_scope(weight_decay=self.weight_decay,
                                           is_training=is_training and self.train_batchnorm)):
            normalized_images = (images / 127.5) - 1.0
            outputs.append(normalized_images)

            desired_endpoints = [
                'layer_2',
                'layer_4',
                'layer_7',
                'layer_14',
                'layer_18'
            ]

            _, endpoints = mobilenet_base(normalized_images)
            for d in desired_endpoints:
                outputs.append(endpoints[d])

        return outputs

    def pretrained_information(self) -> Tuple[Optional[str], Union[None, List, Dict]]:
        return self.pretrained_file, [v for v in tf.global_variables()
                                      if 'MobilenetV2' in v.name and 'renorm' not in v.name]