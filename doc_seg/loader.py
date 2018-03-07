import tensorflow as tf
import os
from threading import Semaphore
import numpy as np
from .input_utils import load_and_resize_image
import tempfile
from scipy.misc import imsave, imread
from PIL import Image

class LoadedModel:
    def __init__(self, model_base_dir, input_dict_key='images', signature_def_key='serving_default',
                 num_parallel_predictions=2):
        if os.path.exists(os.path.join(model_base_dir, 'saved_model.pbtxt')) or \
                os.path.exists(os.path.join(model_base_dir, 'saved_model.pb')):
            model_dir = model_base_dir
        else:
            possible_dirs = os.listdir(model_base_dir)
            model_dir = os.path.join(model_base_dir, max(possible_dirs))  # Take latest export
        print("Loading {}".format(model_dir))

        self.sess = tf.get_default_session()
        loaded_model = tf.saved_model.loader.load(self.sess, ['serve'], model_dir)
        assert 'serving_default' in list(loaded_model.signature_def)

        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def[signature_def_key])
        self._input_tensor = input_dict[input_dict_key]
        self._output_dict = output_dict
        self.sema = Semaphore(num_parallel_predictions)

    def predict(self, input_tensor, prediction_key=None):
        with self.sema:
            if prediction_key:
                desired_output = self._output_dict[prediction_key]
            else:
                desired_output = self._output_dict
            return self.sess.run(desired_output, feed_dict={self._input_tensor: input_tensor})

    def predict_with_tiles(self, filename: str, resized_size: int=None, tile_size: int=500,
                           min_overlap: float=0.2, linear_interpolation: bool=True):

        if resized_size is None or resized_size < 0:
            image_np = imread(filename)
            h, w = image_np.shape[:2]
            b = 1
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
            assert full_output.shape[1] == new_input.shape[1]
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

        result = {k: np.empty([b, h, w] + list(v.shape[3:]), v.dtype) for k, v in all_outputs[0][0].items()}
        if linear_interpolation:
            for k in result.keys():
                assigned_up_to_y = 0
                for y, y_outputs in zip(y_pos, all_outputs):
                    s = list(result[k].shape)
                    tmp = np.zeros([b, tile_size] + s[2:], result[k].dtype)
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
        return result


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}
