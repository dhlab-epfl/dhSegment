import tensorflow as tf
import os
from threading import Semaphore


class Model:
    def __init__(self, model_base_dir, num_parallel_predictions=2):
        possible_dirs = os.listdir(model_base_dir)
        model_dir = os.path.join(model_base_dir, max(possible_dirs))
        print("Loading {}".format(model_dir))

        self.sess = tf.get_default_session()
        loaded_model = tf.saved_model.loader.load(self.sess, ['serve'], model_dir)
        assert 'serving_default' in list(loaded_model.signature_def)

        input_dict, output_dict = _signature_def_to_tensors(loaded_model.signature_def['serving_default'])
        self._input_tensor = input_dict['images']
        self._output_tensor = output_dict['labels']
        self.sema = Semaphore(num_parallel_predictions)

    def predict(self, image_tensor):
        with self.sema:
            return self.sess.run(self._output_tensor, feed_dict={self._input_tensor: image_tensor})


def _signature_def_to_tensors(signature_def):
    g = tf.get_default_graph()
    return {k: g.get_tensor_by_name(v.name) for k,v in signature_def.inputs.items()}, \
           {k: g.get_tensor_by_name(v.name) for k,v in signature_def.outputs.items()}
