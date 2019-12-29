from typing import Dict

import numpy as np
import scipy
from src import svm_classifier

from typing import List
from tqdm import tqdm
import random


def get_nullspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix
    """
    nullspace_basis = scipy.linalg.null_space(W)  # orthogonal basis
    nullspace_basis = nullspace_basis * np.sign(nullspace_basis[0][0])  # handle sign ambiguity
    projection_matrix = nullspace_basis.dot(nullspace_basis.T)

    return projection_matrix


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    P = np.eye(input_dim)
    for v in directions:
        P_v = get_nullspace_projection(v)
        P = P.dot(P_v)

    return P


def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, noise=False, by_class=True, Y_train_main=None,
                             Y_dev_main=None) -> np.ndarray:
    """
    :param classifier_class: the sklearn classifier class (SVM/Perceptron etc.)
    :param cls_params: a dictionary, containing the params for the sklearn classifier
    :param num_classifiers: number of iterations (equivalent to number of dimensions to remove)
    :param input_dim: size of input vectors
    :param is_autoregressive: whether to train the ith classiifer on the data projected to the nullsapces of w1,...,wi-1
    :param min_accuracy: above this threshold, ignore the learned classifier
    :param X_train: ndarray, training vectors
    :param Y_train: ndarray, training labels (protected attributes)
    :param X_dev: ndarray, eval vectors
    :param Y_dev: ndarray, eval labels (protected attributes)
    :param noise: bool, whether to add noise to the vectors
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :return: the debiasing projection
    """

    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))
                            
    P = np.eye(input_dim)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
                            
    if noise:
        print("Adding noise.")
        mean = np.mean(np.abs(X_train))
        mask_train = 0.0075 * (np.random.rand(*X_train.shape) - 0.5)
        X_train_cp += mask_train

    pbar = tqdm(range(num_classifiers))
    for i in pbar:
                            
        clf = svm_classifier.SVMClassifier(classifier_class(**cls_params))

        if by_class:
            cls = np.random.choice(Y_train_main)  # random.choice(main_task_labels) UNCOMMENT FOR EQUAL CHANCE FOR ALL Y
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(x_t.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network(X_train_cp[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev],
                                Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        P_i = get_nullspace_projection(W)
        P = P.dot(P_i)

        if is_autoregressive:
            X_train_cp = X_train_cp.dot(P_i)
            X_dev_cp = X_dev_cp.dot(P_i)

    return P


if __name__ == '__main__':
    X = np.random.rand(5000, 300)
    Y = np.random.rand(5000) < 0.5
    num_classifiers = 15
    classifier_class = None
    input_dim = 300
    is_autoregressive = True
    min_accuracy = 0.5
    noise = False
    random_subset = True
    siamese = True

    P = get_debiasing_projection(classifier_class, None, num_classifiers, input_dim, is_autoregressive, min_accuracy,
                                 X, Y, X, Y, by_class=True)

    print(list(zip(X.dot(P)[0], X.dot(P).dot(P)[0]))[:10])

    print(list(zip(P[0], P.dot(P)[0]))[:10])
    # assert np.allclose(P.dot(P), P)
    # assert np.allclose(P.dot(P), P.T)
    print("yay")
