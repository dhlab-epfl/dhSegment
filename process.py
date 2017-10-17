import argparse
import tensorflow as tf
from doc_seg import model, input, loader
import os
try:
    import better_exceptions
except:
    pass
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def process(input_dir, output_dir, model_dir, resizing_size, gpu):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, visible_device_list=gpu)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
        m = loader.LoadedModel(model_dir)

    os.makedirs(output_dir, exist_ok=True)

    input_filenames = glob(os.path.join(input_dir, '*.jpg')) + \
                      glob(os.path.join(input_dir, '*.png')) + \
                      glob(os.path.join(input_dir, '*.tif')) + \
                      glob(os.path.join(input_dir, '*.jp2'))

    for path in tqdm(input_filenames):
        img = Image.open(path).resize(resizing_size)
        mat = np.asarray(img)
        if len(mat.shape) == 2:
            mat = np.stack([mat, mat, mat], axis=2)
        predictions = m.predict(mat[None], prediction_key='labels')[0]
        plt.imsave(os.path.join(output_dir, os.path.relpath(path, input_dir)), predictions)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-dir", required=True, help="Folder with input images")
    ap.add_argument("-o", "--output-dir", required=True, help="Folder with output results")
    ap.add_argument("-m", "--model-dir", required=True, help="Where the model will be loaded")
    ap.add_argument("-g", "--gpu", type=str, required=True, help="Which GPU to use (0, 1)")
    ap.add_argument("-s", "--size", type=str, required=False, default='1024,688',
                    help="Resizing size of the input image 'w,h'")
    args = vars(ap.parse_args())

    size_str = args.get('size').split(',')
    resizing_size = [int(s) for s in size_str]

    process(args.get('input_dir'), args.get('output_dir'), args.get('model_dir'),
            resizing_size, args.get('gpu'))
