import io
import os
import subprocess
from glob import glob
import pandas as pd
from tqdm import tqdm
from dh_segment.io import PAGE
from .process import extract_lines

CBAD_JAR = './cBAD/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
PP_PARAMS = {'sigma': 1.5, 'low_threshold': 0.2, 'high_threshold': 0.4}


def eval_fn(input_dir: str,
            groudtruth_dir: str,
            output_dir: str=None,
            post_process_params: dict=PP_PARAMS,
            channel_baselines: int=1,
            jar_tool_path: str=CBAD_JAR,
            masks_dir: str=None) -> dict:
    """
    Evaluates a model against the selected set ('groundtruth_dir' contains XML files)

    :param input_dir: Input directory containing probability maps (.npy)
    :param groudtruth_dir: directory containg XML groundtruths
    :param output_dir: output directory for results
    :param post_process_params: parameters form post processing of probability maps
    :param channel_baselines: the baseline class chanel
    :param jar_tool_path: path to cBAD evaluation tool (.jar file)
    :param masks_dir: optional, directory where binary masks of the page are stored (.png)
    :return:
    """

    if output_dir is None:
        output_dir = input_dir

    # Apply post processing and find lines
    for file in tqdm(glob(os.path.join(input_dir, '*.npy'))):
        basename = os.path.basename(file).split('.')[0]
        gt_xml_filename = os.path.join(groudtruth_dir, basename + '.xml')
        gt_page_xml = PAGE.parse_file(gt_xml_filename)

        original_shape = [gt_page_xml.image_height, gt_page_xml.image_width]

        _, _ = extract_lines(file, output_dir, original_shape, post_process_params,
                             channel_baselines=channel_baselines, mask_dir=masks_dir)

    # Create pairs predicted XML - groundtruth XML to be evaluated
    xml_pred_filenames_list = glob(os.path.join(output_dir, '*.xml'))
    xml_filenames_tuples = list()
    for xml_filename in xml_pred_filenames_list:
        basename = os.path.basename(xml_filename)
        gt_xml_filename = os.path.join(groudtruth_dir, basename)

        xml_filenames_tuples.append((gt_xml_filename, xml_filename))

    gt_pages_list_filename = os.path.join(output_dir, 'gt_pages_simple.lst')
    generated_pages_list_filename = os.path.join(output_dir, 'generated_pages_simple.lst')
    with open(gt_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[0] for s in xml_filenames_tuples]))
    with open(generated_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[1] for s in xml_filenames_tuples]))

    # Evaluation using JAVA Tool
    cmd = 'java -jar {} {} {}'.format(jar_tool_path, gt_pages_list_filename, generated_pages_list_filename)
    result = subprocess.check_output(cmd, shell=True).decode()
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        f.write(result)
    parse_score_txt(result, os.path.join(output_dir, 'scores.csv'))

    # Parse results from output of tool
    lines = result.splitlines()
    avg_precision = float(next(filter(lambda l: 'Avg (over pages) P value:' in l, lines)).split()[-1])
    avg_recall = float(next(filter(lambda l: 'Avg (over pages) R value:' in l, lines)).split()[-1])
    f_measure = float(next(filter(lambda l: 'Resulting F_1 value:' in l, lines)).split()[-1])

    print('P {}, R {}, F {}'.format(avg_precision, avg_recall, f_measure))

    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'f_measure': f_measure
    }


def parse_score_txt(score_txt: str, output_csv: str):
    lines = score_txt.splitlines()
    header_ind = next((i for i, l in enumerate(lines)
                       if l == '#P value, #R value, #F_1 value, #TruthFileName, #HypoFileName'))
    final_line = next((i for i, l in enumerate(lines) if i > header_ind and l == ''))
    csv_data = '\n'.join(lines[header_ind:final_line])
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns={k: k.strip() for k in df.columns})
    df['#HypoFileName'] = [os.path.basename(f).split('.')[0] for f in df['#HypoFileName']]
    del df['#TruthFileName']
    df = df.rename(columns={'#P value': 'P', '#R value': 'R', '#F_1 value': 'F_1', '#HypoFileName': 'basename'})
    df = df.reindex(columns=['basename', 'F_1', 'P', 'R'])
    df = df.sort_values('F_1', ascending=True)
    df.to_csv(output_csv, index=False)
