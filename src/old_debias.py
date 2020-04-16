from typing import Dict

import numpy as np
import scipy
from src import classifier

# import svm_classifier

REGRESSION = False
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
                             Y_dev: np.ndarray, noise=False, random_subset=1., by_class=True, Y_train_main=None,
                             Y_dev_main=None) -> np.ndarray:
    """
    :param classifier_class:
    :param num_classifiers:
    :param input_dim:
    :param is_autoregressive:
    :param min_accuracy:
    :param X_train:
    :param Y_train:
    :param X_dev:
    :param Y_dev:
    :return: the debiasing projection
    """

    if by_class and ((Y_train_main is None) or (Y_dev_main is None)): raise Exception()

    P = np.eye(input_dim)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    labels_set = list(set(Y_train.tolist()))
    main_task_labels = list(set(Y_train_main.tolist()))

    if noise:
        print("Adding noise.")
        mean = np.mean(np.abs(X_train))
        mask_train = 0.0075 * (np.random.rand(*X_train.shape) - 0.5)

        X_train_cp += mask_train

    pbar = tqdm(range(num_classifiers))
    for i in pbar:

        x_t, y_t = X_train_cp.copy(), Y_train.copy()

        clf = classifier.SKlearnClassifier(classifier_class(**cls_params))

        idx = np.random.rand(x_t.shape[0]) < random_subset
        x_t = x_t[idx]
        y_t = y_t[idx]

        #if by_class:
        #    cls = np.random.choice(Y_train_main)  # random.choice(main_task_labels) UNCOMMENT FOR EQUAL CHANCE FOR ALL Y
        #    relevant_idx_train = Y_train_main == cls
        #    relevant_idx_dev = Y_dev_main == cls
        #else:
        #    relevant_idx_train = np.ones(x_t.shape[0], dtype=bool)
        #    relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network(x_t, y_t, X_dev_cp, Y_dev)
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        P_i = get_nullspace_projection(W)
        P = P.dot(P_i)

        if is_autoregressive:
            X_train_cp = X_train_cp.dot(P_i)
            X_dev_cp = X_dev_cp.dot(P_i)

    return P

