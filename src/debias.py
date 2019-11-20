import numpy as np
import scipy
from scipy import linalg
from typing import Tuple
import classifier

def get_nullspace_projection(W: np.ndarray) -> np.ndarray:
    """

    :param W: the matrix over its nullspace to project
    :return: the projection matrix
    """
    nullspace_basis = scipy.linalg.null_space(W) # orthogonal basis
    nullspace_basis = nullspace_basis * np.sign(nullspace_basis[0][0]) # handle sign ambiguity
    projection_matrix = nullspace_basis.dot(nullspace_basis.T)

    return projection_matrix


def get_debiasing_projection(classifier_class: classifier.Classifier, num_classifiers: int, input_dim: int, is_autoregressive: bool,
           min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> np.ndarray:

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

    P = np.eye(input_dim)
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()

    for i in range(num_classifiers):

        clf = classifier_class()
        acc = clf.train(X_train_cp, Y_train, X_dev_cp, Y_dev)
        if acc < min_accuracy: continue

        W = clf.get_weights()
        P_i = get_nullspace_projection(W)
        P = P_i.dot(P)

        if is_autoregressive:

            X_train_cp = X_train_cp.dot(P_i)
            X_dev_cp = X_dev_cp.dot(P_i)

    return P