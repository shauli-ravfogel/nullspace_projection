"""
Usage:
  encode_bert_states.py [--input_file=INPUT_FILE] [--output_dir=OUTPUT_DIR] [--split=SPLIT]

Options:
  -h --help                     show this help message and exit
  --input_file=INPUT_FILE       input dir file
  --output_dir=OUTPUT_DIR       write down output file
  --split=SPLIT                 split name

"""

import numpy as np
from docopt import docopt
import torch
from transformers import *
import pickle
from tqdm import tqdm


def read_data_file(input_file):
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_lm():
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer


def tokenize(tokenizer, data):
    tokenized_data = []
    for row in tqdm(data):
        tokens = tokenizer.encode(row['hard_text_untokenized'], add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


def encode_text(model, data):
    all_data_cls = []
    all_data_avg = []
    batch = []
    for row in tqdm(data):
        batch.append(row)
        input_ids = torch.tensor(batch)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].numpy())
        batch = []
    return np.array(all_data_avg), np.array(all_data_cls)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_file = arguments['--input_file']

    out_dir = arguments['--output_dir']
    split = arguments['--split']

    model, tokenizer = load_lm()
    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode",
                                               add_special_tokens=True)])

    data = read_data_file(in_file)
    tokens = tokenize(tokenizer, data)
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

    avg_data, cls_data = encode_text(model, tokens)

    np.save(out_dir + '/' + split + '_avg.npy', avg_data)
    np.save(out_dir + '/' + split + '_cls.npy', cls_data)
