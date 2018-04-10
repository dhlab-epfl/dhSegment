# DocumentSegmentation

DocumentSegmentation allows you to extract content (segment regions) from different type of documents.
It is inspired by the paper [_U-Net: Convolutional Networks for Biomedical Image Segmentation_](https://arxiv.org/pdf/1505.04597.pdf) but uses a pretrained network for encoding.

## Installation and requirements
 See `INSTALL.md` and `environment.yml`.

## Usage
#### Training
* You need to have your training data in a folder containing `images` folder and `labels` folder. The pairs (images, labels) need to have the same name (it is not mandatory to have the same extension file, however we recommend having the label images as `.png` files). 
* The annotated images in `label` folder are (usually) RGB images with the regions to segment annotated with a specific color
* The file containing the classes has the format show below, where each row corresponds to one class (including 'negative' or 'background' class) and each row has 3 values for the 3 RGB values. Of course each class needs to have a different code.
``` class.txt
0 0 0
0 255 0
...
```
* [`sacred`](https://sacred.readthedocs.io/en/latest/quickstart.html) package is used to deal with experiments and trainings. Have a look at the documentation to use it properly.

In order to train a model, you should run `python train.py with <config.json>`

## Demo
This demo shows the usage of dhSegment for page document extraction. It trains a model from scratch (optional) using the [READ-BAD dataset](https://arxiv.org/abs/1705.03311) and the annotations of [pagenet](https://github.com/ctensmeyer/pagenet/tree/master/annotations) (annotator1 is used).
In order to limit memory usage, the images in the dataset we provide have been downsized to have 1M pixels each.

__How to__

1. Get the annotated dataset [here](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/pages.zip), which already contains the folders `images` and `labels` for training, validation and testing set. Unzip it into `model/pages`. 
2. Download the pretrained weights for ResNet :
```
cd pretrained_models/
python download_resnet_pretrained_model.py
cd ..
```

3. You can train the model from scratch with 
    `python train.py with demo/demo_config.json`
    or skip this step and use directly the [provided model](https://github.com/dhlab-epfl/dhSegment/releases/download/v0.2/model.zip) (download and unzip it in `demo/model`)
4. Run `python demo.py`
5. Have a look at the results in `demo/processed_images`


