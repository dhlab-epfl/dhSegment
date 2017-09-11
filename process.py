import argparse
import tensorflow as tf
from doc_seg import model, input, loader
import os
try:
    import better_exceptions
except:
    pass
from tqdm import trange, tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-dir", required=True, help="Folder with input images")
ap.add_argument("-o", "--output-dir", required=True, help="Folder with output results")
ap.add_argument("-m", "--model-dir", required=True, help="Where the model will be loaded")
ap.add_argument("-g", "--gpu", type=str, required=True, help="Which GPU to use (0, 1)")
args = vars(ap.parse_args())


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, visible_device_list=args['gpu'])
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
    m = loader.LoadedModel(args['model_dir'])

os.makedirs(args['output_dir'], exist_ok=True)

input_filenames = glob(os.path.join(args['input_dir'], '*.jpg')) + \
                  glob(os.path.join(args['input_dir'], '*.png')) + \
                  glob(os.path.join(args['input_dir'], '*.tif')) + \
                  glob(os.path.join(args['input_dir'], '*.jp2'))

for path in tqdm(input_filenames):
    img = Image.open(path).resize((416, 608))
    mat = np.asarray(img)
    if len(mat.shape) == 2:
        mat = np.stack([mat, mat, mat], axis=2)
    predictions = m.predict(mat[None], prediction_key='labels')[0]
    plt.imsave(os.path.join(args['output_dir'], os.path.relpath(path, args['input_dir'])), predictions)
