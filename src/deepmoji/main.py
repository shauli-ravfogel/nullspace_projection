"""
Usage:
  main.py [--input_dir=INPUT_DIR] [--output_dir=OUTPUT_DIR]

Options:
  -h --help                     show this help message and exit
  --input_dir=INPUT_DIR         input dir file
  --output_dir=OUTPUT_DIR       write down output file

Script for collecting potential verbs from the aspecutal verbs list.
"""

import numpy as np
from docopt import docopt
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from src import debias


def load_data(path):
    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.npy"]
    labels = [0, 1, 0, 1]
    X, Y = [], []

    for fname, label in zip(fnames, labels):
        print(path + '/' + fname)
        data = np.load(path + '/' + fname)
        for x in data:
            X.append(x)
        for _ in data:
            Y.append(label)

    Y = np.array(Y)
    X = np.array(X)
    X, Y = shuffle(X, Y, random_state=0)
    return X, Y


def find_projection_matrices(X_train, Y_train, X_dev, Y_dev, dim, out_dir):
    num_clfs = [5, 10, 25, 50, 80]
    is_autoregressive = True
    min_acc = 0.
    noise = False

    for n in num_clfs:
        print("num classifiers: {}".format(n))

        clf = LinearSVC
        params = {'max_iter': 3000, 'fit_intercept': True, 'class_weight': "balanced", 'dual': False}

        P_n = debias.get_debiasing_projection(clf, params, n, dim, is_autoregressive, min_acc, X_train, Y_train, X_dev, Y_dev,
                                              noise=noise)
        fname = out_dir + "P.num-clfs={}.npy".format(n)
        np.save(fname, P_n)


if __name__ == '__main__':
    arguments = docopt(__doc__)

    in_dir = arguments['--input_dir']

    out_dir = arguments['--output_dir']

    x_train, y_train = load_data(in_dir + '/train/')
    x_dev, y_dev = load_data(in_dir + '/dev/')
    find_projection_matrices(x_train, y_train, x_dev, y_dev, 300, out_dir)

