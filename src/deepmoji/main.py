import sys
sys.path.append("..")
import debias
import numpy as np
import sklearn
from sklearn.utils import shuffle

def load_data(path):

    fnames = ["neg_neg.npy", "neg_pos.npy", "pos_neg.npy", "pos_pos.py"]
    labels = [0,1,0,1]
    X, Y = [], []

    for fname, label in zip(fnames, labels):

        data = np.load(fname)
        for x in data:
            X.append(x)
        for Y in data:
            Y.append(label)

    X, Y = shuffle(X,Y, random_state = 0)
    X_train, X_dev, y_train, y_dev = \
        sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, X_dev, y_train, y_dev


def find_projection_matrices(X_train,Y_train, X_dev, Y_dev, dim = 2304, attribute = "race"):

    num_clfs = [5, 10, 25, 50, 80]
    is_autoregressive = True
    siamese = False
    reg = "l2"
    min_acc = 0.
    noise = False
    random_subset = False
    regression = False

    for n in num_clfs:
        print("num classifiers: {}".format(n))

        P_n = debias.get_debiasing_projection(None, n, dim, is_autoregressive, min_acc, X_train, Y_train, X_dev, Y_dev,
                                              noise = noise, random_subset = random_subset,
                                              regression = regression, siamese = siamese, siamese_dim = 1)
        fname = "matrices/P.num-clfs={}.attribute={}.npy".foramt(n, attribute)
        with open(fname, "w") as f:
            np.save(f, P_n)

if __name__ == '__main__':

    data_type = sys.argv[1] # race/gender
    path = "/home/nlp/lazary/workspace/thesis/shrink-task-learning/data/processed/emoji_{}".format(data_type)
    X_train, X_dev, y_train, y_dev = load_data(path)
    find_projection_matrices(X_train, y_train, X_dev, y_dev, attribute = data_type)