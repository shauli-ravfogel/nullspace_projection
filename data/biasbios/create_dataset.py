import pickle
import sklearn
from sklearn import model_selection
import tqdm
from collections import Counter
import random
import numpy as np

PROFS = ['professor', 'physician', 'attorney', 'photographer', 'journalist', 'nurse', 'psychologist', 'teacher',
'dentist', 'surgeon', 'architect', 'painter', 'model', 'poet', 'filmmaker', 'software_engineer',
'accountant', 'composer', 'dietitian', 'comedian', 'chiropractor', 'pastor', 'paralegal', 'yoga_teacher',
'dj', 'interior_designer', 'personal_trainer', 'rapper']

PROF2UNIFIED_PROF = {"associate professor": "professor", "assistant professor": "professor", "software engineer": "software_engineer", "psychotherapist": "psychologist", "orthopedic surgeon": "surgeon", "trial lawyer": "attorney","plastic surgeon": "surgeon",  "trial attorney": "attorney", "senior software engineer": "software_engineer", "interior designer": "interior_designer", "certified public accountant": "accountant", "cpa": "accountant", "neurosurgeon": "surgeon", "yoga teacher": "yoga_teacher", "nutritionist": "dietitian", "personal trainer": "personal_trainer", "certified personal trainer": "personal_trainer", "yoga instructor": "yoga_teacher"}


def load_data(fname = "BIOS.pkl"):

        with open(fname, "rb") as f:
        
                data = pickle.load(f)

        print("Number of entries: {}".format(len(data)))
        return data
def preprocess(data):

        # unite similar professions

        for i, data_dict in enumerate(data):
                prof = data_dict["raw_title"].lower()
                data[i]["raw_title"] = PROF2UNIFIED_PROF[prof] if prof in PROF2UNIFIED_PROF else prof

def pickle_data(data, name):
        with open(name+".pickle", "wb") as f:
                pickle.dump(data, f)

def write_to_file(dictionary, name):

        with open(name+".txt", "w", encoding = "utf-8") as f:

                for k,v in dictionary.items():
                        if name == "word2index":
                                print(k,v)
                                print("--------------------")
                        f.write(str(k) + "\t" + str(v) + "\n")

def split_train_dev_test(data, vocab_size):
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
        for entry in tqdm.tqdm(data, total=len(data)):
                gender, prof = entry["gender"].lower(), entry["raw_title"].lower()
                # if prof in PROF2UNITED_PROF: prof = PROF2UNITED_PROF[prof]
                raw, start_index = entry["raw"], entry["start_pos"]
                hard_text = raw[start_index + 1:]
                text_without_gender = entry["bio"]
                all_data.append({"g": gender, "p": prof, "text": raw, "start": start_index, "hard_text": hard_text, "text_without_gender": text_without_gender})

        random.seed(0)
        np.random.seed(0)
        train_dev, test = sklearn.model_selection.train_test_split(all_data, test_size=0.25, random_state=0)
        train, dev = sklearn.model_selection.train_test_split(train_dev, test_size=0.1/0.75, random_state=0)
        print("Train size: {}; Dev size: {}; Test size: {}".format(len(train), len(dev), len(test)))
        pickle_data(train, "train")
        pickle_data(dev, "dev")
        pickle_data(test, "test")
        write_to_file(p2i, "profession2index")
        print(w2i)
        write_to_file(w2i, "word2index")

def main():

        data = load_data()
        preprocess(data)
        split_train_dev_test(data, vocab_size = 250000)

if __name__ == '__main__':
    main()