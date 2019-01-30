#!/usr/bin/env python
# coding: utf-8 

import json
import os
from tqdm import tqdm
import numpy as np
from skimage import transform
from collections import namedtuple
from imageio import imsave, imread
import requests
from itertools import filterfalse
from typing import List, Tuple, Dict
from enum import Enum
import cv2


__author__ = "maudehrmann"

# iiif_password = os.environ["IIIF_PWD"]
iiif_password = ''

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


class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)


def _get_annotations(via_dict: dict, name_file: str) -> dict:
    """
    From VIA json file, get annotations relative to the given `name_file`.

    :param via_dict: VIA annotation output (originally json)
    :param name_file: the file to look for (it can be a iiif path or a file path)
    :return: dict
    """

    # Check that the annotation_dict is a "via_project" file (project export),
    # or a "via_region" file (annotation export)
    if '_via_img_metadata' in via_dict.keys():
        annotation_dict = via_dict['_via_img_metadata']
    else:
        annotation_dict = via_dict

    #  If it looks like a iiif path add "-1"
    if 'http' in name_file:
        key = name_file + "-1"
    else:  # find the key that contains the name_file
        list_keys = list(filterfalse(lambda x: name_file not in x, list(annotation_dict.keys())))
        assert len(list_keys) == 1, "There is more than one key for the file '{} : \n{}'".format(name_file, list_keys)
        key = list_keys[0]

    if key in annotation_dict.keys():
        myannotation = annotation_dict[key]
        if name_file == myannotation['filename']:
            return myannotation['regions']
        else:
            return None


def _compute_reduced_dimensions(x: int, y: int, target_h: int=2000) -> Tuple[int, int]:
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
    print("Collecting working items for {}".format(collection))
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
                target_h, target_w = _compute_reduced_dimensions(x, y)
            else:
                resp_json.raise_for_status()

            regions = _get_annotations(annotations, iiif)

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

    print("Collected {} items.".format(len(working_items)))
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
        img = _getimage_from_iiif(working_item.iiif, 'epfl-team', iiif_password)
        img_resized = transform.resize(
                                img,
                                [working_item.reduced_y, working_item.reduced_x],
                                anti_aliasing=False,
                                preserve_range=True
        )
        imsave(outfile, img_resized.astype(np.uint8))


def _getimage_from_iiif(url, user, pwd):
    img = requests.get(url, auth=(user, pwd))
    return imread(img.content)


def load_annotation_data(via_data_filename: str) -> dict:
    """
    Load the content of via annotation files.

    :param via_data_filename: via annotations json file
    :return: the content of json file containing the region annotated
    """
    with open(via_data_filename, 'r') as f:
        content = json.load(f)

    return content


def export_annotation_dict(annotation_dict: dict, filename: str) -> None:
    """
    Export the annotations to json file.

    :param annotation_dict: VIA annotations
    :param filename: filename to export the data (json file)
    :return:
    """
    with open(filename, 'w') as f:
        json.dump(annotation_dict, f)


def _write_mask(mask: np.ndarray, masks_dir: str, collection: str, image_name: str, label: str) -> None:
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
     Get labels from annotation tool (VIA) settings. Only compatible with VIA v2.0.

    :param annotation_file: manual annotation json file
    :return:  dict (k=label, v=RGB code)
    """
    with open(annotation_file, 'r') as a:
        annotations = json.load(a)

    # todo : this is not generic
    # list(annotations['_via_attributes']['region'][{name_attributes}][{args}]
    label_list = list(annotations['_via_attributes']['region']['Label']['options'])

    # todo: keep a list of colors and assign randomly to label (to avoid hard coding labels/color correspondance)
    # todo: or take it from a config file
    label_color = dict()
    for label in label_list:
        if label == "MYLABEL1":
            label_color[label] = Color.WHITE  # white
        elif label == "MYLABEL2":
            label_color[label] = Color.YELLOW  # yellow
        # etc.
    return label_color


def get_via_attributes_regions(annotation_dict: dict, via_version: int=2) -> List[str]:
    """
    Gets the attributes of the annotataed data.

    :param annotation_dict:
    :param via_version: either 1 or 2 (for VIA v 1.0 or VIA v 2.0)
    :return: A list containing the attributes names§
    """
    if via_version == 1:
        list_categories = list()

        for value in annotation_dict.values():
            regions = value['regions']
            for region in regions.values():
                list_categories += list(region['region_attributes'].keys())

        return list(np.unique(list_categories))
    elif via_version == 2:
        # If project_export is given
        if '_via_attributes' in annotation_dict.keys():
            return list(annotation_dict['_via_attributes']['region'].keys())
        # else if annotation_export is given
        else:
            list_categories = list()
            for value in annotation_dict.values():
                regions = value['regions']
                for region in regions:
                    list_categories += list(region['region_attributes'].keys())

            return list(np.unique(list_categories))
    else:
        raise NotImplementedError


def get_labels_per_attribute(annotation_dict: dict, attribute_regions: List[str], via_version: int=2) -> Tuple:
    """
    For each attribute, get all the possible label variants.

    :param annotation_dict:
    :param attribute_regions:
    :param via_version:
    :return: (unique_labels, dict_labels) : `dict_labels` is a dictionary containing all the labels per attribute

    Usage
    -----

    >>> annotations_dict = load_annotation_data('via_annotations.json')
    >>> list_attributes = get_via_attributes_regions(annotations_dict)
    >>> list_attributes
    >>> ['object', 'text']
    >>>
    >>> unique_labels, dict_labels = get_labels_per_attribute(annotations_dict, list_attributes)
    >>> unique_labels
    >>> [['animal', 'car', ''], ['handwritten', 'typed', '']]
    >>>
    >>> dict_labels
    >>> {'object': ['animal', 'animal', 'car', 'animal', ''], 'text': ['handwritten', '', 'typed', '', 'handwritten']}
    """

    if via_version == 1:
        dict_labels = {ar: list() for ar in attribute_regions}
        for value in annotation_dict.values():
            regions = value['regions']
            for region in regions.values():
                for k, v in region['region_attributes'].items():
                    dict_labels[k].append(v)

        unique_labels = list()
        for ar in attribute_regions:
            unique_labels.append(list(np.unique(dict_labels[ar])))

        return unique_labels, dict_labels
    elif via_version == 2:
        # If project_export is given
        if '_via_attributes' in annotation_dict.keys():
            raise NotImplementedError
        # else if annotation_export is given
        else:
            dict_labels = {ar: list() for ar in attribute_regions}
            for value in annotation_dict.values():
                regions = value['regions']
                for region in regions:
                    for k, v in region['region_attributes'].items():
                        dict_labels[k].append(v)

            unique_labels = list()
            for ar in attribute_regions:
                unique_labels.append(list(np.unique(dict_labels[ar])))

            return unique_labels, dict_labels
    else:
        raise NotImplementedError


def create_masks_v2(masks_dir: str, working_items: List[WorkingItem], annotation_file: str,
                    collection: str, contours_only: bool=False) -> None:
    """
    For each annotation, create a corresponding binary mask and resize it (h = 2000). Only valid for VIA 2.0.
    Several annotations of the same class on the same image produce one image with several masks.

    :param masks_dir: where to output the masks
    :param working_items: infos to work with
    :param annotation_file:
    :param collection:
    :param contours_only: creates the binary masks only for the contours of the object (thickness of contours : 20 px)
    :return: None
    """
    print("Creating masks in {}...".format(masks_dir))

    annotation_summary = dict()

    def resize_and_write_mask(mask_image: np.ndarray, working_item: WorkingItem, label_item: str):
        """
        Resize only if needed (if working_item.reduced != working_item.original)
        :param mask_image:
        :param working_item:
        :param label_item:
        :return:
        """
        if not working_item.reduced_y and not working_item.reduced_x:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)
        elif working_item.reduced_x != working_item.original_x and working_item.reduced_y != working_item.original_y:
            mask_resized = transform.resize(mask_image, [working_item.reduced_y, working_item.reduced_x],
                                            anti_aliasing=False, preserve_range=True, order=0)
            _write_mask(mask_resized, masks_dir, collection, working_item.image_name, label_item)
        else:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)

    for wi in tqdm(working_items, desc="workingItem2mask"):
        labels = []
        label_list = get_labels(annotation_file)
        # the image has no annotation, writing a black mask:
        if not wi.annotations:
            mask = np.zeros([wi.original_y, wi.original_x], np.uint8)
            resize_and_write_mask(mask, wi, None)
            labels.append("nolabel")
        # check all possible labels for the image and create mask:
        else:
            for label in label_list.keys():
                # get annotation corresponding to current label
                # todo : the 'Label' key is not generic
                selected_regions = list(filter(lambda r: r['region_attributes']['Label'] == label, wi.annotations))
                if selected_regions:
                    # create a 0 matrix (black background)
                    mask = np.zeros([wi.original_y, wi.original_x], np.uint8)
                    # add one or several mask for current label
                    # nb: if 2 labels are on the same page, they belongs to the same mask

                    contours_points = list()
                    for sr in selected_regions:

                        if sr['shape_attributes']['name'] == 'rect':
                            x = sr['shape_attributes']['x']
                            y = sr['shape_attributes']['y']
                            w = sr['shape_attributes']['width']
                            h = sr['shape_attributes']['height']

                            contours_points.append(np.array([[x, y],
                                                             [x + w, y],
                                                             [x + w, y + h],
                                                             [x, y + h]
                                                             ]).reshape((-1, 1, 2)))

                        elif sr['shape_attributes']['name'] == 'polygon':
                            contours_points.append(np.stack([sr['shape_attributes']['all_points_x'],
                                                             sr['shape_attributes']['all_points_y']], axis=1)[:, None, :])

                        else:
                            raise NotImplementedError('Mask annotation for shape of type "{}" has not been implemented '
                                                      'yet'.format(sr['shape_attributes']['name']))

                        if contours_only:
                            mask = cv2.polylines(mask, contours_points, True, 255, thickness=15)
                        else:
                            mask = cv2.fillPoly(mask, contours_points, 255)

                    # resize
                    resize_and_write_mask(mask, wi, label)
                    # add to existing labels
                    labels.append(label.strip(' \n').replace(" ", "_").lower())

        # write summary: list of existing labels per image
        annotation_summary[wi.image_name] = labels
        outfile = os.path.join(masks_dir, collection, collection + "-classes.txt")
        fh = open(outfile, 'w')
        for a in annotation_summary:
            fh.write(a + "\t" + str(annotation_summary[a]) + "\n")
        fh.close()

    print("Done.")
    return annotation_summary


def create_masks_v1(masks_dir: str, working_items: List[WorkingItem], collection: str,
                    label_name: str, contours_only: bool=False) -> None:
    """
    For each annotation, create a corresponding binary mask and resize it (h = 2000). Only valid for VIA 1.0.
    Several annotations of the same class on the same image produce one image with several masks.

    :param masks_dir: where to output the masks
    :param working_items: infos to work with
    :param collection:
    :param label_name: name of the label to create mask
    :param contours_only: creates the binary masks only for the contours of the object (thickness of contours : 20 px)
    :return: None
    """

    annotation_summary = dict()

    def resize_and_write_mask(mask_image: np.ndarray, working_item: WorkingItem, label_item: str):
        """
        Resize only if needed (if working_item.reduced != working_item.original)
        :param mask_image:
        :param working_item:
        :param label_item:
        :return:
        """
        if not working_item.reduced_y and not working_item.reduced_x:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)
        elif working_item.reduced_x != working_item.original_x and working_item.reduced_y != working_item.original_y:
            mask_resized = transform.resize(mask_image, [working_item.reduced_y, working_item.reduced_x],
                                            anti_aliasing=False, preserve_range=True, order=0)
            _write_mask(mask_resized, masks_dir, collection, working_item.image_name, label_item)
        else:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)

    for wi in tqdm(working_items, desc="workingItem2mask"):
        labels = []
        # the image has no annotation, writing a black mask:
        if not wi.annotations:
            mask = np.zeros([wi.original_y, wi.original_x], np.uint8)
            resize_and_write_mask(mask, wi, None)
            labels.append("nolabel")
        # check all possible labels for the image and create mask:
        else:
            # get annotation corresponding to current label
            selected_regions = wi.annotations
            if selected_regions:
                # create a 0 matrix (black background)
                mask = np.zeros([wi.original_y, wi.original_x], np.uint8)
                # add one or several mask for current label
                # nb: if 2 labels are on the same page, they belongs to the same mask
                elem_to_iterate = selected_regions.values() if isinstance(selected_regions, dict) else selected_regions

                contours_points = list()
                for sr in elem_to_iterate:
                    if sr['shape_attributes']['name'] == 'rect':
                        x = sr['shape_attributes']['x']
                        y = sr['shape_attributes']['y']
                        w = sr['shape_attributes']['width']
                        h = sr['shape_attributes']['height']

                        contours_points.append(np.array([[x, y],
                                                         [x + w, y],
                                                         [x + w, y + h],
                                                         [x, y + h]
                                                         ]).reshape((-1, 1, 2)))

                        if contours_only:
                            mask = cv2.polylines(mask, contours_points, True, 255, thickness=15)
                        else:
                            mask = cv2.fillPoly(mask, contours_points, 255)

                    elif sr['shape_attributes']['name'] == 'polygon':
                        contours_points.append(np.stack([sr['shape_attributes']['all_points_x'],
                                                         sr['shape_attributes']['all_points_y']], axis=1)[:, None, :])

                        if contours_only:
                            mask = cv2.polylines(mask, contours_points, True, 255, thickness=15)
                        else:
                            mask = cv2.fillPoly(mask, contours_points, 255)

                    elif sr['shape_attributes']['name'] == 'circle':
                        center_point = (sr['shape_attributes']['cx'], sr['shape_attributes']['cy'])
                        radius = sr['shape_attributes']['r']

                        if contours_only:
                            mask = cv2.circle(mask, center_point, radius, 255, thickness=15)
                        else:
                            mask = cv2.circle(mask, center_point, radius, 255, thickness=-1)
                    else:
                        raise NotImplementedError('Mask annotation for shape of type "{}" has not been implemented yet'
                                                  .format(sr['shape_attributes']['name']))

                # resize
                resize_and_write_mask(mask, wi, label_name)
                # add to existing labels
                labels.append(label_name.strip(' \n').replace(" ", "_").lower())

        # write summary: list of existing labels per image
        annotation_summary[wi.image_name] = labels
        outfile = os.path.join(masks_dir, collection, collection + "-classes.txt")
        fh = open(outfile, 'a')
        for a in annotation_summary:
            fh.write(a + "\t" + str(annotation_summary[a]) + "\n")
        fh.close()

    return annotation_summary


# def main(args):
#
#     # read config
#     config_file = args["--config-file"]
#     task = args["--task"]
#     collection = args["--collection"]
#
#     if config_file and os.path.isfile(config_file):
#         print("Found config file: {}".format(os.path.realpath(config_file)))
#         with open(config_file, 'r') as f:
#             config = json.load(f)
#     else:
#         print("Provide a config file")
#
#     annotation_file = config.get("annotation_file")  # manual annotation json file
#     image_url_file = config.get("image_url_file")  # url image list
#     experiments_dir = config.get("experiments_dir")  # output expe
#     masks_dir = config.get("masks_dir")  # output annotation_objects
#     img_out_dir = config.get("img_out_dir")  # re-scaled images
#
#     print("\nGot the following paths:\n"
#           "annotation_file: {}\n"
#           "image_url_file: {}\n"
#           "experiments_dir: {}\n"
#           "masks_dir: {}\n"
#           "img_out_dir: {}\n".format(annotation_file, image_url_file, experiments_dir, masks_dir, img_out_dir)
#           )
#
#     # to test working items loading
#     if task == "test-collect":
#         collect_working_items(image_url_file, annotation_file, collection)
#
#     # scale down and write original images
#     elif task == "original":
#         working_items = collect_working_items(image_url_file, annotation_file, collection)
#         wi_bag = db.from_sequence(working_items, partition_size=100)
#         wi_bag2 = wi_bag.map(scale_down_original, img_out_dir=img_out_dir)
#         with ProgressBar():
#             wi_bag2.compute()
#
#     # create masks
#     elif task == "masks":
#         working_items = collect_working_items(image_url_file, annotation_file, collection)
#         create_masks_v2(masks_dir, working_items, annotation_file, collection)
#


def _get_xywh_from_coordinates(coordinates: np.array) -> Tuple[int, int, int, int]:
    """
    From cooridnates points get x,y, width height
    :param coordinates: (N,2) coordinates (x,y)
    :return: x, y, w, h
    """

    x = np.min(coordinates[:, 0])
    y = np.min(coordinates[:, 1])
    w = np.max(coordinates[:, 0]) - x
    h = np.max(coordinates[:, 1]) - y

    return x, y, w, h


def create_via_region_from_coordinates(coordinates: np.array, region_attributes: dict, type_region: str) -> dict:
    """

    :param coordinates: (N, 2) coordinates (x, y)
    :param region_attributes: dictionary with keys : name of labels, values : values of labels
    :param type_region: via region annotation type ('rect', 'polygon')
    :return: a region in VIA style (dict/json)
    """
    assert type_region in ['rect', 'polygon', 'circle']

    if type_region == 'rect':
        x, y, w, h = _get_xywh_from_coordinates(coordinates)
        shape_atributes = {
            'name': 'rect',
            'height': int(h),
            'width': int(w),
            'x': int(x),
            'y': int(y)
        }
    elif type_region == 'polygon':
        points_x = list(coordinates[:, 0])
        points_y = list(coordinates[:, 1])

        shape_atributes = {
            'name': 'polygon',
            'all_points_x': [int(p) for p in points_x],
            'all_points_y': [int(p) for p in points_y],
        }
    elif type_region == 'circle':
        raise NotImplementedError('The type {} is not supported for the export.'.format(type))

    return {'region_attributes': region_attributes,
            'shape_attributes': shape_atributes}


def create_via_annotation_single_image(img_filename: str, via_regions: List[dict],
                                       file_attributes: dict=None) -> Dict[str, dict]:
    """

    :param img_filename:
    :param via_regions:
    :param file_attributes:
    :return:
    """

    basename = os.path.basename(img_filename)
    file_size = os.path.getsize(img_filename)

    via_key = '{}{}'.format(basename, file_size)

    via_annotation = {
        'file_attributes': file_attributes if file_attributes is not None else dict(),
        'filename': basename,
        'size': file_size,
        'regions': via_regions
    }

    return {via_key: via_annotation}


"""
Example of usage


collection = 'mycollection'
annotation_file = 'via_regions_annotated.json'
image_url_file = 'list_files_image_url.txt'
masks_dir = '/home/project/generated_masks'

working_items = collect_working_items(image_url_file, annotation_file, collection)
create_masks_v2(masks_dir, working_items, annotation_file, collection)
"""

