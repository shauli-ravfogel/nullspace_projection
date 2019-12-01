"""
Usage:
  collect_aspectual.py [--input_dir=INPUT_DIR] [--output_file=OUTPUT_FILE]

Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_file=OUTPUT_FILE     write down output file

Script for collecting potential verbs from the aspecutal verbs list.
"""

import json
from typing import List, Dict

import numpy as np
from docopt import docopt


def read_data_file(input_file: str, main_label, demog_label) -> List[Dict]:
    vecs = np.load(input_file)

    data = []
    for vec in vecs:
        data.append({'vec': vec, 'main_label': main_label, 'demographic_label': demog_label})

    return data


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_fir']

    pos_pos = read_data_file(in_dir + '/pos_pos', 'positive', 'aa')
    pos_neg = read_data_file(in_dir + '/pos_neg', 'positive', 'white')
    neg_pos = read_data_file(in_dir + '/neg_pos', 'negative', 'aa')
    neg_neg = read_data_file(in_dir + '/neg_neg', 'negative', 'white')

    all_data = pos_pos + pos_neg + neg_pos + neg_neg
    np.random.shuffle(all_data)

    out_file = arguments['--output_file']

    with open(out_file, 'w') as f:
        for line in all_data:
            f.write(json.dumps(line) + '\n')
