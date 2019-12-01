"""
Usage:
  collect_aspectual.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR]

Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_dir=OUTPUT_DIR       write down output file

Script for collecting potential verbs from the aspecutal verbs list.
"""

import json
from typing import List, Dict

import numpy as np
from docopt import docopt


def read_data_file(input_file: str):
    vecs = np.load(input_file)

    np.random.shuffle(vecs)

    return vecs[:30000], vecs[30000:32000], vecs[32000:35500]

    # data = []
    # for vec in vecs:
    #     data.append({'vec': vec, 'main_label': main_label, 'demographic_label': demog_label})
    #
    # return data


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_fir']

    out_dir = arguments['--output_dir']

    for split in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
        data = read_data_file(in_dir + '/' + split + '.npy')
        np.save(out_dir + '/' + split + '.npy')

    # pos_pos = read_data_file(in_dir + '/pos_pos')
    # pos_neg = read_data_file(in_dir + '/pos_neg')
    # neg_pos = read_data_file(in_dir + '/neg_pos')
    # neg_neg = read_data_file(in_dir + '/neg_neg')
    #
    # all_data = pos_pos + pos_neg + neg_pos + neg_neg
    # np.random.shuffle(all_data)
    #
    # out_file = arguments['--output_file']
    #
    # with open(out_file, 'w') as f:
    #     for line in all_data:
    #         f.write(json.dumps(line) + '\n')
