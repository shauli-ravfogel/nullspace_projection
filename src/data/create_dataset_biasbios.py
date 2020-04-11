import pickle
import sklearn
from sklearn import model_selection
import tqdm
from collections import Counter
import random
import numpy as np
from typing import List, Dict
import argparse
import spacy

PROFS = ['professor', 'physician', 'attorney', 'photographer', 'journalist', 'nurse', 'psychologist', 'teacher',
'dentist', 'surgeon', 'architect', 'painter', 'model', 'poet', 'filmmaker', 'software_engineer',
'accountant', 'composer', 'dietitian', 'comedian', 'chiropractor', 'pastor', 'paralegal', 'yoga_teacher',
'dj', 'interior_designer', 'personal_trainer', 'rapper']

# a dictionary for uniting similar professions, according to De-Arteaga, Maria, et al. 2019
PROF2UNIFIED_PROF = {"associate professor": "professor", "assistant professor": "professor", "software engineer": "software_engineer", "psychotherapist": "psychologist", "orthopedic surgeon": "surgeon", "trial lawyer": "attorney","plastic surgeon": "surgeon",  "trial attorney": "attorney", "senior software engineer": "software_engineer", "interior designer": "interior_designer", "certified public accountant": "accountant", "cpa": "accountant", "neurosurgeon": "surgeon", "yoga teacher": "yoga_teacher", "nutritionist": "dietitian", "personal trainer": "personal_trainer", "certified personal trainer": "personal_trainer", "yoga instructor": "yoga_teacher"}


def load_data(fname):
        """
        Load the BIOS dataset from De-Arteaga, Maria, et al. 2019
        """
        with open(fname, "rb") as f:

                data = pickle.load(f)

        return data

def preprocess(data: List[dict]):

        # unite similar professions, tokenize
        """
        :param data: List[dictionary]
        :return: none
        changes the data dictionaries in place, uniting similar professions.
        """
    
        for i, data_dict in enumerate(data):
                prof = data_dict["raw_title"].lower()
                data[i]["raw_title"] = PROF2UNIFIED_PROF[prof] if prof in PROF2UNIFIED_PROF else prof

def pickle_data(data, name):
        with open(name+".pickle", "wb") as f:
                pickle.dump(data, f)

def write_to_file(dictionary, name):

        with open(name+".txt", "w", encoding = "utf-8") as f:

                for k,v in dictionary.items():

                        f.write(str(k) + "\t" + str(v) + "\n")

def split_train_dev_test(data: List[dict], output_dir: str, vocab_size: int):
        """
        :param data: list of dictionaries, each containing the biography of a single person
        :param vocab_size: how many words to keep
        :return: none. writes the dataset to files
        """
        g2i, i2g = {"m": 0, "f": 1}, {1: "f", 0: "m"}
        all_profs = list(set([d["raw_title"] for d in data]))
        all_words = []

        for d in data:
                all_words.extend(d["raw"].split(" "))

        words_counter = Counter(all_words)
        common_words, counts = map(list, zip(*words_counter.most_common(vocab_size)))
        common_words = ["<UNK>"] + common_words

        p2i = {p: i for i, p in enumerate(sorted(all_profs))}
        w2i = {w: i for i, w in enumerate(common_words)}

        all_data = []
        nlp = spacy.load("en_core_web_sm") 

        for entry in tqdm.tqdm(data, total=len(data)):
                gender, prof = entry["gender"].lower(), entry["raw_title"].lower()
                raw, start_index = entry["raw"], entry["start_pos"]
                hard_text = raw[start_index + 1:] # the biography without the first line
                hard_text_tokenized =  list(nlp.pipe([hard_text], disable=["tagger", "parser", "ner"]))[0]
                hard_text_tokenized = " ".join([tok.text for tok in hard_text_tokenized])
                text_without_gender = entry["bio"] # the text, with all gendered words and names removed
                all_data.append({"g": gender, "p": prof, "text": raw, "start": start_index, "hard_text": hard_text, "text_without_gender": text_without_gender, "hard_text_tokenized": hard_text_tokenized})

        random.seed(0)
        np.random.seed(0)

        train_prop, dev_prop, test_prop = 0.65, 0.1, 0.25      
        train, dev, test = [], [], []
                
        for prof in all_profs:
            
                relevant_prof = [d for d in all_data if d["p"] == prof]
                relevant_prof_m, relevant_prof_f = [d for d in relevant_prof if d["g"] == "m"],  [d for d in relevant_prof if d["g"] == "f"]
                prof_m_train_dev, prof_m_test = sklearn.model_selection.train_test_split(relevant_prof_m, test_size=0.25, random_state=0)
                prof_m_train, prof_m_dev = sklearn.model_selection.train_test_split(prof_m_train_dev, test_size=0.1/0.75, random_state=0)

                prof_f_train_dev, prof_f_test = sklearn.model_selection.train_test_split(relevant_prof_f, test_size=0.25, random_state=0)
                prof_f_train, prof_f_dev = sklearn.model_selection.train_test_split(prof_f_train_dev, test_size=0.1/0.75, random_state=0)
                
                train.extend(prof_m_train + prof_f_train)
                dev.extend(prof_m_dev + prof_f_dev)
                test.extend(prof_m_test + prof_f_test)
                
        random.shuffle(train)
        random.shuffle(dev)
        random.shuffle(test)
        
        pickle_data(train, output_dir + "train")
        pickle_data(dev, output_dir + "dev")
        pickle_data(test, output_dir + "test")
        write_to_file(p2i, output_dir + "profession2index")
        write_to_file(w2i, output_dir + "word2index")
        write_to_file(g2i, output_dir + "gender2index")

def main():
        parser = argparse.ArgumentParser(
                description='Creating a dataset for the bias-in-bios experiments.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--input-path', dest='input_path', type=str,
                            default='../../data/biasbios/BIOS.pkl',
                            help='path to bios dataset of De-Arteaga, Maria, et al. 2019')
        parser.add_argument('--output-dir', dest='output_dir', type=str,
                            default="../../data/biasbios/",
                            help='output directory.')
        parser.add_argument('--vocab-size', dest='vocab_size', type=int,
                            default=250000,
                            help='how many words to keep.')
        args = parser.parse_args()

        data = load_data(args.input_path)
        preprocess(data)
        split_train_dev_test(data = data, output_dir = args.output_dir, vocab_size = args.vocab_size)

if __name__ == '__main__':
    main() 
