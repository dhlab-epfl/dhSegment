import argparse
import tensorflow as tf
from fc_cnn import model, input, load_model
import os
import better_exceptions
from tqdm import trange, tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-dir", required=True, help="Folder with input images")
ap.add_argument("-o", "--output-dir", required=True, help="Folder with output results")
ap.add_argument("-m", "--model-dir", required=True, help="Where the model will be loaded")
args = vars(ap.parse_args())


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, visible_device_list='0')
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default():
    m = load_model.Model(args['model_dir'])

os.makedirs(args['output_dir'], exist_ok=True)

for path in tqdm(glob(os.path.join(args['input_dir'], '*.jpg'))):
    img = Image.open(path).resize((1024, 688))
    mat = np.asarray(img)
    predictions = m.predict(mat[None])[0]
    plt.imsave(os.path.join(args['output_dir'], os.path.relpath(path, args['input_dir'])), predictions)
