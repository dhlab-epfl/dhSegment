# DocumentSegmentation

DocumentSegmentation allows you to extract content (segment regions) from different type of documents.
It is inspired by the paper [_U-Net: Convolutional Networks for Biomedical Image Segmentation_](https://arxiv.org/pdf/1505.04597.pdf) but uses a pretrained network for encoding.

Available pretrained implementations : 
* VGG-16: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
* Resnet-V1-50 : http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz

### Usage
#### Training
* You need to have your training data in a folder containing `images` folder and `labels` folder. The pairs (images, labels) need to have the same name (it is however not mandatory to have the same extension file, however we recommend having the label images as `.png` files). 
* The annotated images in `label` folder are (usually) RGB images with the regions to segment annotated with a specific color
* The file containing the classes has the format show below, where each row corresponds to one class (including 'negative' or 'background' class) and each row has 3 values for the 3 RGB values. Of course each class needs to have a different code.
``` class.txt
0 0 0
0 255 0
...
```
* There are 3 types of segmentation possible:
  * `CLASSIFICATION` : Segments assigning a class to each pixel of the image
  * `REGRESSION`: Segments using a regression
  * `MULTILABEL` : Segments with the possibility of one pixel belonging to multiple classes (so far only tested with 2 classes)
* It is possible to choose the number of 'up-poolings' and the layers to use in the decoding network. Use `[pretrained_model_name]_selected_levels_upscaling` in `Params` initialization.



## Method

* Train : `python train.py with <config.json>`
* Perform evaluation : `python post_process_evaluation.py -m <model-folders> -p <post-process-configs.json> -t <TASK>`
* Model Selection : `python model_selection.py -e <experiment-folders> -m <metric-to-be-used-for-selection>`



* Generate configs : `python generate_base_configs.py -co /home/docseg_models/generated_configs/ -mo /home/docseg_models/DIBCO/ -c configs/DIBCO/dibco_binarization_config.json configs/DIBCO/dibco_variant2.json`
* Train them : `python worker.py -c /home/docseg_models/generated_configs -f /home/docseg_models/failed_configs`
* Evaluate on validation : `python post_process_evaluation.py -m /home/docseg_models/DIBCO/* -p configs/DIBCO/dibco_post_processing_configs.json -t dibco`
* Find best on validation and apply to test : `python model_selection.py -e /home/docseg_models/DIBCO/*/post_processing/* -m f_measure`
