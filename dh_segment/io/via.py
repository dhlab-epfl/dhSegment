#!/usr/bin/env python
# coding: utf-8 

__author__ = "maudehrmann, solivr"
__license__ = "GPL"

import json
import os
import re
from tqdm import tqdm
import numpy as np
from skimage import transform
from collections import namedtuple
from imageio import imsave, imread
import requests
from PIL import Image
from itertools import filterfalse, chain
from typing import List, Tuple, Dict
import cv2
from . import PAGE


# To define before using the corresponding functions
# iiif_password = os.environ["IIIF_PWD"]
iiif_password = ''


WorkingItem = namedtuple(
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
WorkingItem.__doc__ = """
A container for annotated images.

:param str collection: name of the collection
:param str image_name: name of the image
:param int original_x: original image x size (width)
:param int original_y: original image y size (height)
:param int reduced_x: resized x size
:param int reduced_y: resized y size
:param str iiif: iiif url
:param dict annotations: VIA 'region_attributes'
"""


VIAttribute = namedtuple(
    "VIAttribute", [
        'name',
        'type',
        'options'
    ]
)
VIAttribute.__doc__ = """
A container for VIA attributes.

:param str name: The name of attribute
:param str type: The type of the annotation (dropdown, markbox, ...)
:param list options: The options / labels possible for this attribute.
"""


def parse_via_attributes(via_attributes: dict) -> List[VIAttribute]:
    """
    Parses the VIA attribute dictionary and returns a list of VIAttribute instances

    :param via_attributes: attributes from VIA annotation ('_via_attributes' field)
    :return: list of ``VIAttribute``
    """

    if {'file', 'region'}.issubset(set(via_attributes.keys())):
        via_attributes = via_attributes['region']

    list_attributes = list()
    for k, v in via_attributes.items():
        if v['type'] == 'text':
            print('WARNING : Please do not use text type for attributes because it is more prone to errors/typos which '
                  'can make the parsing fail. Use instead "checkbox", "dropdown" or "radio" with defined options.')
            options = None
        else:
            options = list(v['options'].keys())

        list_attributes.append(VIAttribute(k,
                                           v['type'],
                                           options))

    return list_attributes


def get_annotations_per_file(via_dict: dict, name_file: str) -> dict:
    """
    From VIA json content, get annotations relative to the given `name_file`.

    :param via_dict: VIA annotations content (originally json)
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
    else:
        # find the key that contains the name_file
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
    target_w = int(target_h * ratio)
    return target_h, target_w


def _collect_working_items_from_local_images(via_annotations: dict, images_dir: str, collection_name: str) \
        -> List[WorkingItem]:
    """
    Given VIA annotation input, collect all info on `WorkingItem` object, when images come from local files

    :param via_annotations: via_annotations: via annotations ('regions' field)
    :param images_dir: directory where to find the images
    :param collection_name: name of the collection
    :return:
    """

    def _formatting(name_id: str) -> str:
        name_id = re.sub('.jpg\d*', '.jpg', name_id)
        name_id = re.sub('.png\d*', '.png', name_id)
        return name_id

    def _get_image_shape_without_loading(filename: str) -> Tuple[int, int]:
        image = Image.open(filename)
        shape = image.size
        image.close()
        return shape

    working_items = list()

    for key, v in tqdm(via_annotations.items()):
        filename = _formatting(key)

        absolute_filename = os.path.join(images_dir, filename)
        shape_image = _get_image_shape_without_loading(absolute_filename)

        regions = v['regions']

        if regions:
            wk_item = WorkingItem(collection=collection_name,
                                  image_name=filename.split('.')[0],
                                  original_x=shape_image[0],
                                  original_y=shape_image[1],
                                  reduced_x=None,
                                  reduced_y=None,
                                  iiif=None,
                                  annotations=regions)

            working_items.append(wk_item)

    return working_items


def _collect_working_items_from_iiif(via_annotations: dict, collection_name: str, iiif_user='my-team') -> dict:
    """
    Given VIA annotation input, collect all info on `WorkingItem` object, when the images come from IIIF urls

    :param via_annotations: via_annotations: via annotations ('regions' field)
    :param collection_name: name of the collection
    :param iiif_user: user param for requests.Session().get()
    :return:
    """

    working_items = list()
    session = requests.Session()

    for key, v in tqdm(via_annotations.items()):
        iiif_url = v['filename']

        image_name = os.path.basename(iiif_url.split('/full/full/')[0])

        # get image dimensions
        iiif_json = iiif_url.replace("default.jpg", "info.json")
        resp_json = session.get(iiif_json, auth=(iiif_user, iiif_password))
        if resp_json.status_code == requests.codes.ok:
            y = resp_json.json()['height']
            x = resp_json.json()['width']
            # target_h, target_w = _compute_reduced_dimensions(x, y)
            target_h, target_w = None, None
        else:
            x, y, target_w, target_h = None, None, None, None
            resp_json.raise_for_status()

        regions = v['regions']

        if regions:
            wk_item = WorkingItem(collection=collection_name,
                                  image_name=image_name.split('.')[0],
                                  original_x=x,
                                  original_y=y,
                                  reduced_x=target_w,
                                  reduced_y=target_h,
                                  iiif=iiif_url,
                                  annotations=regions)

            working_items.append(wk_item)

    return working_items


def collect_working_items(via_annotations: dict, collection_name: str, images_dir: str=None,
                          via_version: int=2) -> List[WorkingItem]:
    """
    Given VIA annotation input, collect all info on `WorkingItem` object.
    This function will take care of separating images from local files and images from IIIF urls.

    :param via_annotations: via annotations ('regions' field)
    :param images_dir: directory where to find the images
    :param collection_name: name of the collection
    :param via_version: version of the VIA tool used to produce the annotations (1 or 2)
    :return: list of `WorkingItem`
    """

    via_annotations_v2 = via_annotations.copy()
    if via_version == 1:
        for key, value in via_annotations_v2.items():
            list_regions = list()
            for v_region in value['regions'].values():
                list_regions.append(v_region)
            via_annotations_v2[key]['regions'] = list_regions

    local_annotations = {k: v for k, v in via_annotations_v2.items() if 'http' not in k}
    url_annotations = {k: v for k, v in via_annotations_v2.items() if 'http' in k}

    working_items = list()
    if local_annotations:
        assert images_dir is not None
        working_items += _collect_working_items_from_local_images(local_annotations, images_dir, collection_name)
    if url_annotations:
        working_items += _collect_working_items_from_iiif(url_annotations, collection_name)

    return working_items


def _scale_down_original(working_item, img_out_dir: str) -> None:
    """
    Copy and reduce original image files.

    :param img_out_dir: where to put the downscaled images
    :param working_item: dict of `WorkingItems`
    :return: None
    """

    def _getimage_from_iiif(url, user, pwd):
        img = requests.get(url, auth=(user, pwd))
        return imread(img.content)

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


def load_annotation_data(via_data_filename: str, only_img_annotations: bool=False, via_version: int=2) -> dict:
    """
    Load the content of via annotation files.

    :param via_data_filename: via annotations json file
    :param only_img_annotations: load only the images annotations ('_via_img_metadata' field)
    :param via_version:
    :return: the content of json file containing the region annotated
    """

    with open(via_data_filename, 'r', encoding='utf8') as f:
        content = json.load(f)
    if via_version == 2:
        assert '_via_img_metadata' in content.keys(), "The file is not a valid VIA project export."

        if only_img_annotations:
            return content['_via_img_metadata']
        else:
            return content
    else:
        return content


def export_annotation_dict(annotation_dict: dict, filename: str) -> None:
    """
    Export the annotations to json file.

    :param annotation_dict: VIA annotations
    :param filename: filename to export the data (json file)
    :return:
    """
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(annotation_dict, f)


def get_via_attributes(annotation_dict: dict, via_version: int=2) -> List[VIAttribute]:
    """
    Gets the attributes of the annotated data and returns a list of `VIAttribute`.

    :param annotation_dict: json content of the VIA exported file
    :param via_version: either 1 or 2 (for VIA v 1.0 or VIA v 2.0)
    :return: A list containing VIAttributes
    """

    if via_version == 1:

        list_attributes = [list(region['region_attributes'].keys())
                           for value in annotation_dict.values()
                           for region in value['regions'].values()]

        # Find options
        unique_attributes = list(np.unique(list(chain.from_iterable(list_attributes))))

        dict_labels = {rgn_att: list() for rgn_att in unique_attributes}
        for value in annotation_dict.values():
            regions = value['regions']
            for region in regions.values():
                for k, v in region['region_attributes'].items():
                    dict_labels[k].append(v)

    elif via_version == 2:

        if '_via_attributes' in annotation_dict.keys():  # If project_export is given
            return parse_via_attributes(annotation_dict['_via_attributes'])

        else:  # else if annotation_export is given

            list_attributes = [list(region['region_attributes'].keys())
                               for value in annotation_dict.values()
                               for region in value['regions']]

            # Find options
            unique_attributes = list(np.unique(list(chain.from_iterable(list_attributes))))

            dict_labels = {rgn_att: list() for rgn_att in unique_attributes}
            for value in annotation_dict.values():
                regions = value['regions']
                for region in regions:
                    for k, v in region['region_attributes'].items():
                        dict_labels[k].append(v)

    else:
        raise NotImplementedError

    # Instantiate VIAttribute objects
    viattribute_list = list()
    for attribute, options in dict_labels.items():

        if all(isinstance(opt, str) for opt in options):
            viattribute_list.append(VIAttribute(name=attribute,
                                                type=None,
                                                options=list(np.unique(options))))

        elif all(isinstance(opt, dict) for opt in options):
            viattribute_list.append(VIAttribute(name=attribute,
                                                type=None,
                                                options=list(np.unique(list(chain.from_iterable(options))))))

        else:
            raise NotImplementedError
    return viattribute_list


def _draw_mask(via_region: dict, mask: np.array, contours_only: bool=False) -> np.array:
    """

    :param via_region: region to draw (in VIA format)
    :param mask: image mask to draw on
    :param contours_only: if `True`, draws only the contours of the region, if `False`, fills the region
    :return: the drawn mask
    """

    shape_attributes_dict = via_region['shape_attributes']

    if shape_attributes_dict['name'] == 'rect':
        x = shape_attributes_dict['x']
        y = shape_attributes_dict['y']
        w = shape_attributes_dict['width']
        h = shape_attributes_dict['height']

        contours = np.array([[x, y],
                             [x + w, y],
                             [x + w, y + h],
                             [x, y + h]
                             ]).reshape((-1, 1, 2))

        mask = cv2.polylines(mask, [contours], True, 255, thickness=15) if contours_only \
            else cv2.fillPoly(mask, [contours], 255)

    elif shape_attributes_dict['name'] == 'polygon':
        contours = np.stack([shape_attributes_dict['all_points_x'],
                             shape_attributes_dict['all_points_y']], axis=1)[:, None, :]

        mask = cv2.polylines(mask, [contours], True, 255, thickness=15) if contours_only \
            else cv2.fillPoly(mask, [contours], 255)

    elif shape_attributes_dict['name'] == 'circle':
        center_point = (shape_attributes_dict['cx'], shape_attributes_dict['cy'])
        radius = shape_attributes_dict['r']

        mask = cv2.circle(mask, center_point, radius, 255, thickness=15) if contours_only \
            else cv2.circle(mask, center_point, radius, 255, thickness=-1)

    elif shape_attributes_dict['name'] == 'polyline':
        contours = np.stack([shape_attributes_dict['all_points_x'],
                             shape_attributes_dict['all_points_y']], axis=1)[:, None, :]

        mask = cv2.polylines(mask, [contours], False, 255, thickness=15)

    else:
        raise NotImplementedError(
            'Mask annotation for shape of type "{}" has not been implemented yet'
                .format(shape_attributes_dict['name']))

    return mask


def _write_mask(mask: np.ndarray, masks_dir: str, collection: str, image_name: str, label: str) -> None:
    """
    Save a mask with filename containing 'label'.

    :param mask: mask b&w image (H, W)
    :param masks_dir: directory to output mask
    :param collection: name of the collection
    :param image_name: name of the image
    :param label: label of the mask
    :return:
    """

    outdir = os.path.join(masks_dir, collection, image_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    label = label.strip(' \n').replace(" ", "_").lower() if label is not None else 'nolabel'
    outfile = os.path.join(outdir, image_name + "-mask-" + label + ".png")
    imsave(outfile, mask.astype(np.uint8))


def create_masks(masks_dir: str, working_items: List[WorkingItem], via_attributes: List[VIAttribute],
                 collection: str, contours_only: bool=False) -> dict:
    """
    For each annotation, create a corresponding binary mask and resize it (h = 2000). Only valid for VIA 2.0.
    Several annotations of the same class on the same image produce one image with several masks.

    :param masks_dir: where to output the masks
    :param working_items: infos to work with
    :param via_attributes: VIAttributes computed by ``get_via_attributes`` function.
    :param collection: name of the nollection
    :param contours_only: creates the binary masks only for the contours of the object (thickness of contours : 20 px)
    :return: annotation_summary, a dictionary containing a list of labels per image
    """

    def resize_and_write_mask(mask_image: np.ndarray, working_item: WorkingItem, label_item: str) -> None:
        """
        Resize only if needed (if working_item.reduced != working_item.original)

        :param mask_image: mask image to write
        :param working_item: `WorkingItem` object
        :param label_item: label name to append to filename
        :return:
        """

        if not working_item.reduced_y and not working_item.reduced_x:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)

        elif working_item.reduced_x != working_item.original_x and working_item.reduced_y != working_item.original_y:
            mask_resized = transform.resize(mask_image,
                                            [working_item.reduced_y, working_item.reduced_x],
                                            anti_aliasing=False,
                                            preserve_range=True,
                                            order=0)
            _write_mask(mask_resized, masks_dir, collection, working_item.image_name, label_item)

        else:
            _write_mask(mask_image, masks_dir, collection, working_item.image_name, label_item)
    # -------------------

    print("Creating masks in {}...".format(masks_dir))

    annotation_summary = dict()

    for wi in tqdm(working_items, desc="workingItem2mask"):
        labels = list()

        # the image has no annotation, writing a black mask:
        if not wi.annotations:
            mask = np.zeros([wi.original_y, wi.original_x], np.uint8)
            resize_and_write_mask(mask, wi, None)
            labels.append("nolabel")

        # check all possible labels for the image and create mask:
        else:
            for attribute in via_attributes:
                for option in attribute.options:
                    # get annotations that have the current attribute
                    selected_regions = list(filter(lambda r: attribute.name in r['region_attributes'].keys(),
                                                   wi.annotations))
                    # get annotations that have the current attribute and option
                    if selected_regions:
                        selected_regions = list(filter(lambda r: r['region_attributes'][attribute.name] == option,
                                                       selected_regions))
                    else:
                        continue

                    if selected_regions:
                        # create a 0 matrix (black background)
                        mask = np.zeros([wi.original_y, wi.original_x], np.uint8)

                        # nb: if 2 labels are on the same page, they belongs to the same mask
                        for sr in selected_regions:
                            mask = _draw_mask(sr, mask, contours_only)

                        label = '{}-{}'.format(attribute.name, option).lower()
                        resize_and_write_mask(mask, wi, label)
                        # add to existing labels
                        labels.append(label)

        # write summary: list of existing labels per image
        annotation_summary[wi.image_name] = labels
        outfile = os.path.join(masks_dir, collection, collection + "-classes.txt")
        with open(outfile, 'a') as fh:
            for a in annotation_summary:
                fh.write(a + "\t" + str(annotation_summary[a]) + "\n")

    print("Done.")
    return annotation_summary


def _get_coordinates_from_xywh(via_regions: List[dict]) -> List[np.array]:
    """
    From VIA region dictionaries, get the coordinates array (N,2) of the annotations

    :param via_regions:
    :return:
    """
    list_coordinates_regions = list()
    for region in via_regions:
        shape_attributes_dict = region['shape_attributes']
        if shape_attributes_dict['name'] == 'rect':
            x = shape_attributes_dict['x']
            y = shape_attributes_dict['y']
            w = shape_attributes_dict['width']
            h = shape_attributes_dict['height']

            coordinates = np.array([[x, y],
                                    [x + w, y],
                                    [x + w, y + h],
                                    [x, y + h]
                                    ])
            list_coordinates_regions.append(coordinates)
        elif shape_attributes_dict['name'] == 'polygon':
            coordinates = np.stack([shape_attributes_dict['all_points_x'],
                                    shape_attributes_dict['all_points_y']], axis=1)
            list_coordinates_regions.append(coordinates)
        elif shape_attributes_dict['name'] == 'polyline':
            coordinates = np.stack([shape_attributes_dict['all_points_x'],
                                    shape_attributes_dict['all_points_y']], axis=1)
            list_coordinates_regions.append(coordinates)
        else:
            raise NotImplementedError(
                "This method has not been implemenetd yet for {}".format(shape_attributes_dict['name']))

    return list_coordinates_regions


# EXPORT
# ------

def _get_xywh_from_coordinates(coordinates: np.array) -> Tuple[int, int, int, int]:
    """
    From coordinates points get x,y, width, height

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
    Formats coordinates to a VIA region (dict).

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
    Returns a dictionary item {key: annotation} in VIA format to further export to .json file

    :param img_filename: path to the image
    :param via_regions: regions in VIA format (output from ``create_via_region_from_coordinates``)
    :param file_attributes: file attributes (usually None)
    :return: dictionary item with key and annotations in VIA format
    """
    if 'http' in img_filename:
        basename = img_filename
        file_size = -1
    else:
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


# PAGE CONVERSION
# ---------------

def convert_via_region_page_text_region(working_item: WorkingItem, structure_label: str) -> PAGE.Page:
    """

    :param working_item:
    :param structure_label:
    :return:
    """

    # TODO : this is not yet generic because we're missing the automatic detection of the structure label

    region_coordinates = _get_coordinates_from_xywh(working_item.annotations)

    page = PAGE.Page(image_filename=working_item.image_name + 'jpg',
                     image_width=working_item.original_x,
                     image_height=working_item.original_y,
                     graphic_regions=[
                         PAGE.TextRegion(coords=PAGE.Point.array_to_point(coords),
                                         custom_attribute='structure{{type:{};}}'.format(structure_label))
                         for coords in region_coordinates])
    return page


# def convert_via_region_to_text_region(via_regions: List[dict], structure_label: str) -> PAGE.TextRegion:
#     """
#
#     :param via_region:
#     :param structure_label:
#     :return:
#     """
#
#     # TODO : this is not yet generic because we're missing the automatic detection of the structure label
#
#     region_coordinates = _get_coordinates_from_xywh(working_item.annotations)
#
#     page = PAGE.Page(image_filename=working_item.image_name + 'jpg',
#                      image_width=working_item.original_x,
#                      image_height=working_item.original_y,
#                      graphic_regions=[
#                          PAGE.TextRegion(coords=PAGE.Point.array_to_point(coords),
#                                          custom_attribute='structure{{type:{};}}'.format(structure_label))
#                          for coords in region_coordinates])
#     return page


"""
Example of usage


collection = 'mycollection'
annotation_file = 'via_sample.json'
masks_dir = '/home/project/generated_masks'
images_dir = './my_images'

# Load all the data in the annotation file (the file may be an exported project or an export of the annotations)
via_data = load_annotation_data(annotation_file)

# In the case of an exported project file, you can set ``only_img_annotations=True`` to get only
# the region annotations
via_annotations = load_annotation_data(annotation_file, only_img_annotations=True)

# Collect the annotated regions
working_items = collect_working_items(via_annotations, collection, images_dir)

# Collect the attributes and options
if '_via_attributes' in via_data.keys():
    list_attributes = parse_via_attributes(via_data['_via_attributes'])
else:
    list_attributes = get_via_attributes(via_annotations)

# Create one mask per option per attribute
create_masks(masks_dir, wi,via_attributes, collection)
"""


"""
Content of a via_project exported file

{'_via_attributes': {
    ...
    },
 '_via_img_metadata': {
    ...
    },
 '_via_settings': {
    'core': {
        'buffer_size': 18,
        'default_filepath': '',
        'filepath': {}
    },
    'project': {
        'name': 'via_project_7Feb2019_10h7m'
    },
    'ui': {
        'annotation_editor_fontsize': 0.8,
        'annotation_editor_height': 25,
        'image': {
            'region_label': 'region_id',
            'region_label_font': '10px Sans'
        },
        'image_grid': {
            'img_height': 80,
            'rshape_fill': 'none',
            'rshape_fill_opacity': 0.3,
            'rshape_stroke': 'yellow',
            'rshape_stroke_width': 2,
            'show_image_policy': 'all',
            'show_region_shape': True
        },
        'leftsidebar_width': 18
    }
 }
}

"""

"""
"_via_attributes": {
    "region": {
        "attribute1": {
            "type":"text",
            "description":"",
            "default_value":""
        },
        "attribute2": {
            "type":"dropdown",
            "description":"",
            "options": {
                "op1":"",
                "op2":""
                },
            "default_options":{}
        },
        "attribute3": {
            "type":"checkbox",
            "description":"",
            "options": {
                "op1":"",
                "op2":""
            },
            "default_options":{}
        },
        "attribute 4": {
            "type":"radio",
            "description":"",
            "options": {
                "op1":"",
                "op2":""
            },
            "default_options":{}
        }
    },
    "file":{}
}

"""

"""
'_via_img_metadata': {
    'image_filename1.jpg2209797': {
        'file_attributes': {},
        'filename': 'image_filename1.jpg',
        'regions':
            [{
                'region_attributes': {
                    'attribute1': {
                        'op1': True,
                        'op2': True
                    },
                    'attribute 2': 'label1',
                    'attribute 3': 'op1'
                },
                'shape_attributes': {
                    'height': 2277,
                    'name': 'rect',
                    'width': 1541,
                    'x': 225,
                    'y': 458
                }
            },
            {
                'region_attributes': {
                    'attribute 4': 'op1',
                    'attribute 1': {},
                    'attribute 2': 'label1',
                    'attribute 3': 'op2'
                },
                'shape_attributes': {
                    'height': 2255,
                    'name': 'rect',
                    'width': 1554,
                    'x': 1845,
                    'y': 476
                }
            }],
            'size': 2209797},
    'https://libimages.princeton.edu/loris/pudl0001/5138415/00000011.jp2/full/full/0/default.jpg-1': {
        'file_attributes': {},
        'filename': 'https://libimages.princeton.edu/loris/pudl0001/5138415/00000011.jp2/full/full/0/default.jpg',
        'regions':
            [{
                'region_attributes': {
                    'attribute 4': 'op2',
                    'attribute 1': {
                        'op1': True
                    },
                    'attribute 2': 'label3',
                    'attribute 3': 'op1'
                },
                'shape_attributes': {
                    'height': 1026,
                    'name': 'rect',
                    'width': 1430,
                    'x': 145,
                    'y': 525
                }
            },
            {
                'region_attributes': {
                    'attribute 4': 'op2',
                    'attribute 1': {
                        'op1': True},
                    'attribute 2': 'label 3 ',
                    'attribute 3': 'op1',
                },
                'shape_attributes': {
                    'all_points_x': [2612, 2498, 2691, 2757, 2962, 3034, 2636],
                    'all_points_y': [5176, 5616, 5659, 5363, 5375, 5110, 5122],
                    'name': 'polygon'
                }
            },
            {
                'region_attributes': {
                    'attribute 4': 'op2',
                    'attribute 1': {
                        'op1': True},
                    'attribute 2': 'label 3 ',
                    'attribute 3': 'op1',
                },
                'shape_attributes': {
                    'cx': 2793,
                    'cy': 881,
                    'name': 'circle',
                    'r': 524
                }
            },
            {
                'region_attributes': {
                    'attribute 4': 'op1',
                    'attribute 1': {
                        'op2': True},
                    'attribute 2': 'label1',
                    'attribute 3': 'op2',
                },
                'shape_attributes': {
                    'all_points_x': [3246, 5001],
                    'all_points_y': [422, 380],
                    'name': 'polyline'
                }
            }],
        'size': -1
    }
}
"""
