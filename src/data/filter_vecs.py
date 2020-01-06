import argparse
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import tqdm
import string
import codecs
import json
import sys
from typing import List

def get_excluded_words():
    with codecs.open('../lists/gender_specific_full.json') as f:
        gender_specific = json.load(f)
    with codecs.open('../lists/definitional_pairs.json') as f:
        definitional_pairs = json.load(f)
    with codecs.open('../lists/equalize_pairs.json') as f:
        equalize_pairs = json.load(f)

    exclude_words = []
    for pair in definitional_pairs + equalize_pairs:
        exclude_words.append(pair[0])
        exclude_words.append(pair[1])

    exclude_words = list(set(exclude_words).union(set(gender_specific)))
    return exclude_words

def get_names():

        with open("../lists/first_names_clean.txt", "r") as f:
                lines = f.readlines()
        
        names = [l.strip().lower() for l in lines]
        return names
        
def load_model(fname, binary):
    model = KeyedVectors.load_word2vec_format(fname, binary=binary)
    vecs = model.vectors
    words = list(model.vocab.keys())

    return model, vecs, words


def has_punct(w):
    if any([c in string.punctuation for c in w]):
        return True
    return False


def has_digit(w):
    if any([c in '0123456789' for c in w]):
        return True
    return False

def save_in_word2vec_format(vecs: np.ndarray, words: np.ndarray, fname: str):


    with open(fname, "w") as f:

        f.write(str(len(vecs)) + " " + "300" + "\n")
        for i, (v,w) in tqdm.tqdm(enumerate(zip(vecs, words))):

            vec_as_str = " ".join([str(x) for x in v])
            f.write(w + " " + vec_as_str + "\n")

def filter_vecs(vecs: np.ndarray, words: np.ndarray, keep_gendered: bool, keep_names: bool):

    filtered = []
    gendered_words = []

    exlucded_words = get_excluded_words()
    first_names = set(get_names())

    for i, (v,w) in tqdm.tqdm(enumerate(zip(vecs, words))):
        
        if (w in first_names) and not keep_names:
                continue
                 
        if w in exlucded_words:
            gendered_words.append((w,v))
            if not keep_gendered:           
                continue
                
        if len(w) >= 20:
            continue
        if has_digit(w):
            continue
        if '_' in w:
            p = [has_punct(subw) for subw in w.split('_')]
            if not any(p):
                filtered.append((w,v))
            continue
        if has_punct(w):
            continue

        filtered.append((w,v))

    words, vecs = zip(*filtered)
    vecs = np.array(vecs)
    words = list(words)

    words_gendered, vecs_gendered = zip(*gendered_words)
    vecs_gendered = np.array(vecs_gendered)
    words_gendered = list(words_gendered)

    return (vecs, words), (words_gendered, vecs_gendered)

def save_voc(voc: List[str], path: str):

        with open(path, "w", encoding = "utf-8") as f:
        
                for w in voc:
                
                        f.write(w + "\n")
                        

def main():

    parser = argparse.ArgumentParser(description='Filtering pretrained word embeddings for word2vec debiasing experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-path', dest='input_path', type=str,
                        default='../../data/embeddings/glove.42B.300d.txt',
                        help='path to embeddings; NOTE: ASSUMES LOWERCASE!')
    parser.add_argument('--output-dir', dest='output_dir', type=str,
                        default="../../data/embeddings/",
                        help='output directory.')
    parser.add_argument('--top-k', dest='top_k', type=int,
                        default= 150000,
                        help='how many word vectors to keep.')  
    parser.add_argument('--keep-inherently-gendered', dest='keep_gendered', type=bool,
                        default=True,
                        help='if true, keeps inherently gendered words such as father, mother, queen, king.')
    parser.add_argument('--keep-names', dest='keep_names', type=bool,
                        default=True,
                        help='if true, keeps private names')                           
    args = parser.parse_args()
                                                  
    model, vecs, words = load_model(args.input_path, binary = False)
    (vecs, words), (words_gendered, vecs_gendered) = filter_vecs(vecs, words, args.keep_gendered, args.keep_names)
    save_in_word2vec_format(vecs[:args.top_k], words[:args.top_k], args.output_dir + "vecs.filtered.txt")
    save_in_word2vec_format(vecs_gendered, words_gendered, args.output_dir + "vecs.gendered.txt")
    save_voc(words[:args.top_k], args.output_dir + "voc.txt")
    
if __name__ == '__main__':
    main()
