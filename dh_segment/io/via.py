#!/usr/bin/env python
# coding: utf-8 

"""
Script with CLI to process annotation data produced with VGG Image Annotation (VIA) tool:
- scale down original images
- parse VIA annotation (json)
- create images with masks
(cf. http://www.robots.ox.ac.uk/~vgg/software/via/; done on VIA 2.0.0)


Usage:
    via.py --task=<t> --collection=<c> --config-file=<cf>

Options:
    --collection<collection> document collection to work with
    --task=<t> task to do: 'original' to downscale original images or 'masks' to create masks.
    --config-file=<cf>  configuration file

"""


import docopt
import json
import sys
import os
from tqdm import tqdm
import numpy as np
from skimage import transform
from collections import namedtuple
from imageio import imsave, imread
import logging
import requests
from typing import List, Tuple

from dask.diagnostics import ProgressBar
import dask.bag as db


__author__ = "maudehrmann"

iiif_password = os.environ["IIIF_PWD"]

logger = logging.getLogger(__name__)

WorkingItem = namedtuple(  # TODO:
    "WorkingItem", [
        'collection',
        'image_name',
        'original_x',
        'original_y',
        'reduced_x',
        'reduced_y',
        'iiif',
        'annotations'
    ]
)


def init_logger(logger, log_level, log_file):
    """Initialise the logger."""
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    )

    if log_file is not None:
        fh = logging.FileHandler(filename=log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Logger successfully initialised")

    return logger


def get_annotations(annotations_dict: dict, iiif_url: str) -> dict:
    """
    From VIA json file, get annotations relative to the given `iiif_url`.

    :param annotations_dict: VIA annotation output (originally json)
    :param iiif_url: the file to look for
    :return: dict
    """
    k = iiif_url + "-1"
    if k in annotations_dict['_via_img_metadata']:
        myannotation = annotations_dict['_via_img_metadata'][k]
        if iiif_url == myannotation['filename']:
            return myannotation['regions']
        else:
            return None


def compute_reduced_dimensions(x: int, y: int, target_h: int=2000) -> Tuple[int, int]:
    """
    Compute new dimensions with height set to `target_h`.

    :param x: height
    :param y: width
    :param target_h: target height
    :return: tuple
    """
    ratio = y / x
    target_w = int(target_h*ratio)
    return target_h, target_w


def collect_working_items(image_url_file: List[str], annotation_file: str, collection: str) -> List[WorkingItem]:
    """
    Given VIA annotation input, collect all info on `WorkingItem` object.

    :param image_url_file: file listing IIIF URLs files
    :param annotation_file: VIA json file, output of manual annotation
    :param collection: target collection to consider
    :return: list of `WorkingItem`
    """
    logger.info("Collecting working items for {}".format(collection))
    working_items = []
    session = requests.Session()

    # load manual annotation json data
    with open(annotation_file, 'r') as a:
        annotations = json.load(a)

    # iterate over image IIIF URLs and build working items
    with open(image_url_file) as current_file:
        lines = current_file.readlines()
        for line in tqdm(lines, desc='URL2WorkingItem'):
            x = None
            y = None
            target_h = None
            target_w = None

            # line is e.g. 'https://myserver.ch/iiif_project/image-name/full/full/0/default.jpg'
            basename = "https://myserver.ch/iiif_project/"  # todo: update, or even pass as param
            iiif = line.strip("\n")

            # get image-name
            image_name = line.split(basename)[1].split("/full/full/0/default.jpg")[0]

            # get image dimensions
            iiif_json = iiif.replace("default.jpg", "info.json")
            resp_image = session.get(iiif, auth=('epfl-team', iiif_password))  # need to request image first
            resp_json = session.get(iiif_json, auth=('epfl-team', iiif_password))
            if resp_json.status_code == requests.codes.ok:
                x = resp_json.json()['height']
                y = resp_json.json()['width']
                target_h, target_w = compute_reduced_dimensions(x, y)
            else:
                resp_json.raise_for_status()

            regions = get_annotations(annotations, iiif)

            wk_item = WorkingItem(
                collection,
                image_name,
                x,
                y,
                target_h,
                target_w,
                iiif,
                regions
            )
            working_items.append(wk_item)

    logger.info("Collected {} items.".format(len(working_items)))
    return working_items


def scale_down_original(working_item, img_out_dir: str) -> None:
    """
    Copy and reduce original image files.

    :param img_out_dir: where to put the downscaled images
    :param working_item: dict of `WorkingItems`
    :return: None
    """
    image_set_dir = os.path.join(img_out_dir, working_item.collection, "images")
    if not os.path.exists(image_set_dir):
        try:
            os.makedirs(image_set_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
            pass

    outfile = os.path.join(image_set_dir, working_item.image_name + "_ds.png")
    if not os.path.isfile(outfile):
        img = getimage_from_iiif(working_item.iiif, 'epfl-team', iiif_password)
        img_resized = transform.resize(
                                img,
                                [working_item.reduced_x, working_item.reduced_y],
                                anti_aliasing=False,
                                preserve_range=True
        )
        imsave(outfile, img_resized.astype(np.uint8))


def getimage_from_iiif(url, user, pwd):
    img = requests.get(url, auth=(user, pwd))
    return imread(img.content)


def write_mask(mask: np.ndarray, masks_dir: str, collection: str, image_name: str, label: str) -> None:
    """
    Save a mask with filename containing 'label'.

    :param mask:
    :param masks_dir:
    :param collection:
    :param image_name:
    :param label:
    :return:
    """
    outdir = os.path.join(masks_dir, collection, image_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    label = label.strip(' \n').replace(" ", "_").lower() if label is not None else 'nolabel'
    outfile = os.path.join(outdir, image_name + "-mask-" + label + ".png")
    # if not os.path.isfile(outfile):
    imsave(outfile, mask.astype(np.uint8))


def get_labels(annotation_file: str) -> dict:
    """
     Get labels from annotation tool (VIA) settings.

    :param annotation_file: manual annotation json file
    :return:  dict (k=label, v=RGB code)
    """
    with open(annotation_file, 'r') as a:
        annotations = json.load(a)

    label_list = list(annotations['_via_attributes']['region']['Label']['options'])

    # todo: keep a list of colors and assign randomly to label (to avoid hard coding labels/color correspondance)
    # todo: or take it from a config file
    label_color = dict()
    for label in label_list:
        if label == "MYLABEL1":
            label_color[label] = (255, 255, 255)  # white
        elif label == "MYLABEL2":
            label_color[label] = (255, 255, 0)  # yellow
        # etc.
    return label_color


def create_masks(masks_dir: str, working_items: List[WorkingItem], annotation_file: str, collection: str) -> None:
    """
    For each annotation, create a corresponding binary mask and resize it (h = 2000).
    Several annotations of the same class on the same image produce one image with several masks.

    :param masks_dir: where to output the masks
    :param working_items: infos to work with
    :param annotation_file:
    :param collection:
    :return: None
    """
    logger.info("Creating masks in {}...".format(masks_dir))

    annotation_summary = dict()

    for wi in tqdm(working_items, desc="workingItem2mask"):
        labels = []
        label_list = get_labels(annotation_file)
        # the image has no annotation, writing a black mask:
        if not wi.annotations:
            mask = np.zeros([wi.original_x, wi.original_y], np.uint8)
            mask_resized = transform.resize(mask, [wi.reduced_x, wi.reduced_y], anti_aliasing=False,
                                            preserve_range=True, order=0)
            write_mask(mask_resized, masks_dir, collection, wi.image_name, None)
            labels.append("nolabel")
        # check all possible labels for the image and create mask:
        else:
            for label in label_list.keys():
                # get annotation corresponding to current label
                selected_regions = list(filter(lambda r: r['region_attributes']['Label'] == label, wi.annotations))
                if selected_regions:
                    # create a 0 matrix (black background)
                    mask = np.zeros([wi.original_x, wi.original_y], np.uint8)
                    # add one or several mask for current label
                    # nb: if 2 labels are on the same page, they belongs to the same mask
                    for sr in selected_regions:
                        x = sr['shape_attributes']['x']
                        y = sr['shape_attributes']['y']
                        w = sr['shape_attributes']['width']
                        h = sr['shape_attributes']['height']
                        # project region(s) on the mask (binary b/w)
                        mask[y:y + h, x:x + w] = 255

                    # resize
                    mask_resized = transform.resize(mask, [wi.reduced_x, wi.reduced_y], anti_aliasing=False,
                                                    preserve_range=True, order=0)
                    # write
                    write_mask(mask_resized, masks_dir, collection, wi.image_name, label)
                    # add to existing labels
                    labels.append(label.strip(' \n').replace(" ", "_").lower())

        # write summary: list of existing labels per image
        annotation_summary[wi.image_name] = labels
        outfile = os.path.join(masks_dir, collection, collection + "-classes.txt")
        fh = open(outfile, 'w')
        for a in annotation_summary:
            fh.write(a + "\t" + str(annotation_summary[a]) + "\n")
        fh.close()

    logger.info("Done.")
    return annotation_summary


def main(args):

    # logger
    global logger
    init_logger(logger, logging.INFO, log_file=None)

    # read config
    config_file = args["--config-file"]
    task = args["--task"]
    collection = args["--collection"]

    if config_file and os.path.isfile(config_file):
        logger.info("Found config file: {}".format(os.path.realpath(config_file)))
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        logging.info("Provide a config file")

    annotation_file = config.get("annotation_file")  # manual annotation json file
    image_url_file = config.get("image_url_file")  # url image list
    experiments_dir = config.get("experiments_dir")  # output expe
    masks_dir = config.get("masks_dir")  # output annotation_objects
    img_out_dir = config.get("img_out_dir")  # re-scaled images

    logger.info("\nGot the following paths:\n"
                "annotation_file: {}\n"
                "image_url_file: {}\n"
                "experiments_dir: {}\n"
                "masks_dir: {}\n"
                "img_out_dir: {}\n".format(annotation_file, image_url_file, experiments_dir, masks_dir, img_out_dir)
                )

    # to test working items loading
    if task == "test-collect":
        collect_working_items(image_url_file, annotation_file, collection)

    # scale down and write original images
    elif task == "original":
        working_items = collect_working_items(image_url_file, annotation_file, collection)
        wi_bag = db.from_sequence(working_items, partition_size=100)
        wi_bag2 = wi_bag.map(scale_down_original, img_out_dir=img_out_dir)
        with ProgressBar():
            wi_bag2.compute()

    # create masks
    elif task == "masks":
        working_items = collect_working_items(image_url_file, annotation_file, collection)
        create_masks(masks_dir, working_items, annotation_file, collection)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    main(arguments)
