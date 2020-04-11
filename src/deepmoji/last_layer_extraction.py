"""
Usage:
  last_layer_extraction.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR] [--model=MODEL]

Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_dir=OUTPUT_DIR       write down output file
  --model=MODEL                 the allennlp model file

Collecting the last hidden layer states of an MLP model (the layer which will be used for debiasing)
"""

from docopt import docopt

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import numpy as np

import torch

from src.framework.models.deep_moji_model import DeepMojiModel
from src.framework.dataset_readers.deep_moji_reader import DeepMojiReader


def transform_vec(in_vec, model):
    module = model.extract_module('scorer')

    x = torch.tensor(in_vec).float()
    # running the first layer of the mlp and the activation function
    for layer in module[:2]:
        x = layer(x)
    return x


def read_data_file(input_file: str):
    vecs = np.load(input_file)
    return vecs


def calculate_vectors(in_dir, out_dir):
    for split in ['train', 'dev', 'test']:
        for vectors in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
            data = read_data_file(in_dir + '/' + split + '/' + vectors + '.npy')
            transformed_vec = transform_vec(data, archive_model)
            np.save(out_dir + '/' + split + '/' + vectors + '.npy', transformed_vec)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_dir']
    model_path = arguments['--model']

    out_dir = arguments['--output_dir']

    archive_model = load_archive(model_path)
    module = archive_model.extract_module('scorer')

    calculate_vectors(in_dir, out_dir)
