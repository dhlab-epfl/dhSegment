import tensorflow as tf
import os
from threading import Semaphore
import numpy as np
import tempfile
from imageio import imsave, imread
from typing import List

_original_shape_key = 'original_shape'


class LoadedModel:
    """
    Loads an exported dhSegment model

    :param model_base_dir: the model directory i.e. containing `saved_model.{pb|pbtxt}`. If not, it is assumed to \
    be a TF exporter directory, and the latest export directory will be automatically selected.
    :param predict_mode: defines the input/output format of the prediction output
    (see `.predict()`)
    :param num_parallel_predictions: limits the number of concurrent calls of `predict` to avoid Out-Of-Memory \
    issues if predicting on GPU
    """

    def __init__(self, model_base_dir, predict_mode='filename', num_parallel_predictions=2):
        if os.path.exists(os.path.join(model_base_dir, 'saved_model.pbtxt')) or \
                os.path.exists(os.path.join(model_base_dir, 'saved_model.pb')):
            model_dir = model_base_dir
        else:
            possible_dirs = os.listdir(model_base_dir)
            model_dir = os.path.join(model_base_dir, max(possible_dirs))  # Take latest export
        print("Loading {}".format(model_dir))

        if predict_mode == 'filename':
            input_dict_key = 'filename'
            signature_def_key = 'serving_default'
        elif predict_mode == 'filename_original_shape':
            input_dict_key = 'filename'
            signature_def_key = 'resized_output'
        elif predict_mode == 'image':
            input_dict_key = 'image'
            signature_def_key = 'from_image:serving_default'
        elif predict_mode == 'image_original_shape':
            input_dict_key = 'image'
            signature_def_key = 'from_image:resized_output'
        elif predict_mode == 'resized_images':
            input_dict_key = 'resized_images'
            signature_def_key = 'from_resized_images:serving_default'
        # elif predict_mode == 'batch_filenames':
        #     input_dict_key = 'filenames'
        #     signature_def_key = 'serving_default'
        else:
            raise NotImplementedError
        self.predict_mode = predict_mode

        self.sess = tf.get_default_session()
        loaded_model = tf.saved_model.loader.load(self.sess, ['serve'], model_dir)
        assert 'serving_default' in list(loaded_model.signature_def)

        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def[signature_def_key])
        assert input_dict_key in input_dict.keys(), "{} not present in input_keys, " \
                                                    "possible values: {}".format(input_dict_key, input_dict.keys())
        self._input_tensor = input_dict[input_dict_key]
        self._output_dict = output_dict

        if predict_mode == 'resized_images':
            # This node is not defined in this specific run-mode as there is no original image
            del self._output_dict['original_shape']
        # elif predict_mode == 'batch_filenames':
        #     self._batch_size = input_dict['batch_size']

        self.sema = Semaphore(num_parallel_predictions)

    def predict(self, input_tensor, prediction_key: str=None):
        """
        Performs the prediction from the loaded model according to the prediction mode. \n
        Prediction modes:

        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+
        | `prediction_mode`           | `input_tensor`                                | Output prediction dictionnary        | Comment                                                                                           |
        +=============================+===============================================+======================================+===================================================================================================+
        | `filename`                  | Single filename string                        | `labels`, `probs`, `original_shape`  | Loads the image, resizes it, and predicts                                                         |
        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+
        | `filename_original_shape`   | Single filename string                        | `labels`, `probs`                    | Loads the image, resizes it, predicts and scale the output to the original resolution of the file |
        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+
        | `image`                     | Single input image [1,H,W,3] float32 (0..255) | `labels`, `probs`, `original_shape`  | Resizes the image, and predicts                                                                   |
        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+
        | `image_original_shape`      | Single input image [1,H,W,3] float32 (0..255) | `labels`, `probs`                    | Resizes the image, predicts, and scale the output to the original resolution of the input         |
        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+
        | `image_resized`             | Single input image [1,H,W,3] float32 (0..255) | `labels`, `probs`                    | Predicts from the image input directly                                                            |
        +-----------------------------+-----------------------------------------------+--------------------------------------+---------------------------------------------------------------------------------------------------+

        :param input_tensor: a single input whose format should match the prediction mode
        :param prediction_key: if not `None`, will returns the value of the corresponding key of the output dictionnary \
        instead of the full dictionnary
        :return: the prediction output
        """
        with self.sema:
            if prediction_key:
                desired_output = self._output_dict[prediction_key]
            else:
                desired_output = self._output_dict
            return self.sess.run(desired_output, feed_dict={self._input_tensor: input_tensor})

    # def batch_predict(self, input_tensor: List[str], batch_size: int=8, prediction_key: str=None):
    #     """
    #     Performs the prediction from the loaded model according to the prediction mode. \n
    #     Prediction modes:
    #
    #     +-------------------+--------------------------+------------------------------------------------------+-------------------------------------------+
    #     | `prediction_mode` | `input_tensor`           | Output prediction dictionnary                        | Comment                                   |
    #     +===================+==========================+======================================================+===========================================+
    #     | `batch_filenames` | List of filename strings | `labels`, `probs`, `original_shape`, `resized_shape` | Loads the image, resizes it, and predicts |
    #     +-------------------+--------------------------+------------------------------------------------------+-------------------------------------------+
    #
    #     :param input_tensor: a batch input whose format should match the prediction mode
    #     :param batch_size: batch size for batch prediction
    #     :param prediction_key: if not `None`, will returns the value of the corresponding key of the output dictionary \
    #     instead of the full dictionary
    #     :return: the prediction output
    #     """
    #     assert len(input_tensor) <= batch_size, "Length of input should be smaller or equal to batch size."
    #     with self.sema:
    #         if prediction_key:
    #             desired_output = self._output_dict[prediction_key]
    #         else:
    #             desired_output = self._output_dict
    #
    #         g = tf.get_default_graph()
    #         _init_op = g.get_operation_by_name('dataset_init')
    #
    #         _, predictions = self.sess.run([_init_op, desired_output], feed_dict={self._input_tensor: input_tensor,
    #                                                                               self._batch_size: batch_size})
    #
    #         return predictions

    def predict_with_tiles(self, filename: str, resized_size: int=None, tile_size: int=500,
                           min_overlap: float=0.2, linear_interpolation: bool=True):

        # TODO this part should only happen if self.predict_mode == 'resized_images'

        if resized_size is None or resized_size < 0:
            image_np = imread(filename)
            h, w = image_np.shape[:2]
            batch_size = 1
        else:
            raise NotImplementedError
        assert h > tile_size, w > tile_size
        # Get x and y coordinates of beginning of tiles and compute prediction for each tile
        y_step = np.ceil((h - tile_size) / (tile_size * (1 - min_overlap)))
        x_step = np.ceil((w - tile_size) / (tile_size * (1 - min_overlap)))
        y_pos = np.round(np.arange(y_step + 1) / y_step * (h - tile_size)).astype(np.int32)
        x_pos = np.round(np.arange(x_step + 1) / x_step * (w - tile_size)).astype(np.int32)

        all_outputs = list()
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i, y in enumerate(y_pos):
                inside_list = list()
                for j, x in enumerate(x_pos):
                    filename_tile = os.path.join(tmpdirname, 'tile{}{}.png'.format(i, j))
                    imsave(filename_tile, image_np[y:y + tile_size, x:x + tile_size])
                    inside_list.append(self.predict(filename_tile))#, prediction_key='probs'))
                all_outputs.append(inside_list)

        def _merge_x(full_output, assigned_up_to, new_input, begin_position):
            assert full_output.shape[1] == new_input.shape[1], \
                "Shape full output is {}, but shape new_input is {}".format(full_output.shape[1], new_input.shape[1])
            overlap_size = assigned_up_to - begin_position
            normal_part_size = new_input.shape[2] - overlap_size
            assert normal_part_size > 0
            full_output[:, :, assigned_up_to:assigned_up_to + normal_part_size] = new_input[:, :, overlap_size:]
            if overlap_size > 0:
                weights = np.arange(0, overlap_size) / overlap_size
                full_output[:, :, begin_position:assigned_up_to] = (1 - weights)[:, None] * full_output[:, :,
                                                                                            begin_position:assigned_up_to] + \
                                                                   weights[:, None] * new_input[:, :, :overlap_size]

        def _merge_y(full_output, assigned_up_to, new_input, begin_position):
            assert full_output.shape[2] == new_input.shape[2]
            overlap_size = assigned_up_to - begin_position
            normal_part_size = new_input.shape[1] - overlap_size
            assert normal_part_size > 0
            full_output[:, assigned_up_to:assigned_up_to + normal_part_size] = new_input[:, overlap_size:]
            if overlap_size > 0:
                weights = np.arange(0, overlap_size) / overlap_size
                full_output[:, begin_position:assigned_up_to] = (1 - weights)[:, None, None] * full_output[:,
                                                                                               begin_position:assigned_up_to] + \
                                                                weights[:, None, None] * new_input[:, :overlap_size]

        result = {k: np.empty([batch_size, h, w] + list(v.shape[3:]), v.dtype) for k, v in all_outputs[0][0].items()
                  if k != _original_shape_key}  # do not try to merge 'original_shape' content...
        if linear_interpolation:
            for k in result.keys():
                assigned_up_to_y = 0
                for y, y_outputs in zip(y_pos, all_outputs):
                    s = list(result[k].shape)
                    tmp = np.zeros([batch_size, tile_size] + s[2:], result[k].dtype)
                    assigned_up_to_x = 0
                    for x, output in zip(x_pos, y_outputs):
                        _merge_x(tmp, assigned_up_to_x, output[k], x)
                        assigned_up_to_x = x + tile_size
                    _merge_y(result[k], assigned_up_to_y, tmp, y)
                    assigned_up_to_y = y + tile_size
        else:
            for k in result.keys():
                for y, y_outputs in zip(y_pos, all_outputs):
                    for x, output in zip(x_pos, y_outputs):
                        result[k][:, y:y + tile_size, x:x + tile_size] = output[k]

        result[_original_shape_key] = np.array([h, w], np.uint)
        return result


class BatchLoadedModel:
    """
        Loads an exported dhSegment model (batch prediction)

        :param model_base_dir: the model directory i.e. containing `saved_model.{pb|pbtxt}`. If not, it is assumed to \
        be a TF exporter directory, and the latest export directory will be automatically selected.
        :param predict_mode: defines the input/output format of the prediction output
        (see `.batch_predict()`)
        :param num_parallel_predictions: limits the number of concurrent calls of `predict` to avoid Out-Of-Memory \
        issues if predicting on GPU
        """

    def __init__(self, model_base_dir, predict_mode='batch_filenames', num_parallel_predictions=2):
        if os.path.exists(os.path.join(model_base_dir, 'saved_model.pbtxt')) or \
                os.path.exists(os.path.join(model_base_dir, 'saved_model.pb')):
            model_dir = model_base_dir
        else:
            possible_dirs = os.listdir(model_base_dir)
            model_dir = os.path.join(model_base_dir, max(possible_dirs))  # Take latest export
        print("Loading {}".format(model_dir))

        if predict_mode == 'batch_filenames':
            input_dict_key = 'filenames'
            signature_def_key = 'serving_default'
        else:
            raise NotImplementedError
        self.predict_mode = predict_mode

        self.sess = tf.get_default_session()
        loaded_model = tf.saved_model.loader.load(self.sess, ['serve'], model_dir)
        assert 'serving_default' in list(loaded_model.signature_def)

        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def[signature_def_key])
        assert input_dict_key in input_dict.keys(), "{} not present in input_keys, " \
                                                    "possible values: {}".format(input_dict_key, input_dict.keys())
        self._input_tensor = input_dict[input_dict_key]
        self._output_dict = output_dict

        self._batch_size = input_dict['batch_size']

        self.sema = Semaphore(num_parallel_predictions)

    def init_prediction(self, input_filenames: List[str], batch_size: int=8):
        """

        :param input_tensor:
        :param batch_size:
        :param prediction_key:
        :return:
        """

        g = tf.get_default_graph()
        _init_op = g.get_operation_by_name('dataset_init')

        _ = self.sess.run(_init_op, feed_dict={self._input_tensor: input_filenames,
                                               self._batch_size: batch_size})

    def predict_next_batch(self, prediction_key: str=None):
        """
        Performs the prediction from the loaded model according to the prediction mode. \n
        Prediction modes:

        +-------------------+--------------------------+------------------------------------------------------+-------------------------------------------+
        | `prediction_mode` | `input_tensor`           | Output prediction dictionnary                        | Comment                                   |
        +===================+==========================+======================================================+===========================================+
        | `batch_filenames` | List of filename strings | `labels`, `probs`, `original_shape`, `resized_shape` | Loads the image, resizes it, and predicts |
        +-------------------+--------------------------+------------------------------------------------------+-------------------------------------------+

        :param input_tensor: a batch input whose format should match the prediction mode
        :param batch_size: batch size for batch prediction
        :param prediction_key: if not `None`, will returns the value of the corresponding key of the output dictionary \
        instead of the full dictionary
        :return: the prediction output
        """
        with self.sema:
            if prediction_key:
                desired_output = self._output_dict[prediction_key]
            else:
                desired_output = self._output_dict

            predictions = self.sess.run(desired_output)

            return predictions


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}
