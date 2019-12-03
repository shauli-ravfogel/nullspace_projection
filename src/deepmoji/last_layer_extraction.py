import sys
import json
from collections import defaultdict, Counter
import itertools

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.models.ensemble import Ensemble

from tqdm import tqdm
import numpy as np

import torch

from src.framework.models.deep_moji_model import DeepMojiModel
from src.framework.dataset_readers.deep_moji_reader import DeepMojiReader


archive_model = load_archive('src/allen_logs/deep_moji_balanced2/model.tar.gz')

module = archive_model.extract_module('scorer')


def transform_vec(in_vec, model):
    module = model.extract_module('scorer')

    x = torch.tensor(in_vec).float()
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


calculate_vectors('../data/emoji_sent_race/', '../data/emoji_sent_race_mlp')