import os
from doc_seg_datasets import PAGE
from glob import glob
import numpy as np
from ..utils import load_pickle
import shutil
import tempfile
import subprocess
from scipy.misc import imread, imresize, imsave
import cv2


CBAD_JAR = '/home/datasets/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
COPIED_GT_PAGE_DIR_NAME = 'gt_page'

# TODO GT files should be moved to a standard folder during dataset generation
GT_DIR = '/home/datasets/cBAD/Baseline_Competition_Simple_Documents/Train'


def cbad_evaluate_folder(output_folder: str, validation_dir: str, verbose=False,
                        debug_folder=None, jar_path: str = CBAD_JAR) -> dict:
    """

    :param output_folder: contains the *.pkl output files of the post-processing
    :param validation_labels_dir: xml files and lst
    :param jar_path:
    :return:
    """

    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # Copy xml gt PAGE to output directory in order to use java evaluation tool
    with tempfile.TemporaryDirectory() as tmpdirname:
        target_page_dir = os.path.abspath(os.path.join(tmpdirname, COPIED_GT_PAGE_DIR_NAME))
        copy_gt_page_to_exp_directory(target_page_dir, GT_DIR)

        filenames_processed = glob(os.path.join(output_folder, '*.pkl'))

        xml_filenames_list = list()
        for filename in filenames_processed:
            basename = os.path.basename(filename).split('.')[0]
            gt_page = PAGE.parse_file(os.path.join(target_page_dir,
                                                   '{}.xml'.format(os.path.basename(filename).split('.')[0])))

            contours, lines_mask = load_pickle(filename)
            ratio = (gt_page.image_height/lines_mask.shape[0], gt_page.image_width/lines_mask.shape[1])
            xml_filename = os.path.join(tmpdirname, basename + '.xml')
            PAGE.save_baselines(xml_filename, contours, ratio)

            xml_filenames_list.append((get_gt_page_filename(xml_filename), xml_filename))

            if debug_folder is not None:
                img = imread(os.path.join(validation_dir, 'images', basename+'.jpg'))
                img = imresize(img, lines_mask.shape[:2])
                img = cv2.polylines(img.copy(), contours, False, (255, 0, 0), thickness=5)
                imsave(os.path.join(debug_folder, basename+'.jpg'), img)

        gt_pages_list_filename = os.path.join(tmpdirname, 'gt_pages.lst')
        generated_pages_list_filename = os.path.join(tmpdirname, 'generated_pages.lst')
        with open(gt_pages_list_filename, 'w') as f:
            f.writelines('\n'.join([s[0] for s in xml_filenames_list]))
        with open(generated_pages_list_filename, 'w') as f:
            f.writelines('\n'.join([s[1] for s in xml_filenames_list]))

        # Run command line evaluation tool
        cwd = os.getcwd()
        os.chdir(os.path.dirname(gt_pages_list_filename))
        cmd = 'java -jar {} {} {}'.format(jar_path, gt_pages_list_filename, generated_pages_list_filename)
        result = subprocess.check_output(cmd, shell=True).decode()
        if debug_folder is not None:
            with open(os.path.join(debug_folder, 'scores.txt'), 'w') as f:
                f.write(result)
        lines = result.splitlines()
        avg_precision = float(next(filter(lambda l: 'Avg (over pages) P value:' in l, lines)).split()[-1])
        avg_recall = float(next(filter(lambda l: 'Avg (over pages) R value:' in l, lines)).split()[-1])
        f_measure = float(next(filter(lambda l: 'Resulting F_1 value:' in l, lines)).split()[-1])
        os.chdir(cwd)
        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f_measure': f_measure
        }


def get_gt_page_filename(exported_xml):
    return os.path.join(os.path.dirname(exported_xml), COPIED_GT_PAGE_DIR_NAME,
                        '{}.xml'.format(os.path.basename(exported_xml).split('.')[0]))


def copy_gt_page_to_exp_directory(target_page_dir: str, gt_dir: str) -> None:
    """
    Copies the original gt PAGE xml file to the postprocessing directory to be able to launch the
    java tool for evaluation
    :param target_page_dir: postprocessing dir
    :param gt_dir: original PAGE directories (usually organized this way : gt_dir/<collection_id>/page/*.xml)
    :return:
    """
    os.makedirs(target_page_dir, exist_ok=True)
    if len(glob(os.path.join(target_page_dir, '*'))) > 0:
        print('Already copied PAGE files.')
        return

    xml_filenames_list = glob(os.path.join(gt_dir, '**', 'page', '*.xml'))

    for filename in xml_filenames_list:
        directory, basename = os.path.split(filename)
        acronym = directory.split(os.path.sep)[-2].split('_')[0]
        new_filenname = '{}_{}.xml'.format(acronym, basename.split('.')[0])
        shutil.copy(filename, os.path.join(target_page_dir, new_filenname))
