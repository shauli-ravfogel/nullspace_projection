"""
Usage:
  collect_aspectual.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR]

Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_dir=OUTPUT_DIR       write down output file

Script for collecting potential verbs from the aspecutal verbs list.
"""

import numpy as np
from docopt import docopt


def read_data_file(input_file: str):
    vecs = np.load(input_file)

    np.random.shuffle(vecs)

    return vecs[:40000], vecs[40000:42000], vecs[42000:44000]


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_dir']

    out_dir = arguments['--output_dir']

    for split in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
        train, dev, test = read_data_file(in_dir + '/' + split + '.npy')
        for split_dir, data in zip(['train', 'dev', 'test'], [train, dev, test]):
            np.save(out_dir + '/' + split_dir + '/' + split + 'npy', data)
