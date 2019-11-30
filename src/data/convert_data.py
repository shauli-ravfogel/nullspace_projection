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


def read_data_file(input_file: str, vocab: dict, main_label, demog_label) -> List[Dict]:
    with open(input_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip().split() for x in lines]

    data = []
    for line in lines:
        text = [vocab[x] for x in line]
        data.append({'text': text, 'main_label': main_label, 'demographic_label': demog_label})

    return data


def read_vocab(in_f):
    with open(in_f, 'r') as f:
        vocab = f.readlines()
        vocab = [x.strip() for x in vocab]
        vocab = dict(enumerate(vocab))
    return vocab


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_fir']
    vocab = read_vocab(in_dir + '/vocab')

    pos_pos = read_data_file(in_dir + '/pos_pos', vocab, 'positive', 'aa')
    pos_neg = read_data_file(in_dir + '/pos_neg', vocab, 'positive', 'white')
    neg_pos = read_data_file(in_dir + '/neg_pos', vocab, 'negative', 'aa')
    neg_neg = read_data_file(in_dir + '/neg_neg', vocab, 'negative', 'white')

    all_data = pos_pos + pos_neg + neg_pos + neg_neg
    np.random.shuffle(all_data)

    out_file = arguments['--output_file']

    with open(out_file, 'w') as f:
        for line in all_data:
            f.write(json.dumps(line) + '\n')
