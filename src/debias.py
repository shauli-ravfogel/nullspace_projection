from typing import Dict

import numpy as np
import scipy
from src import svm_classifier

REGRESSION = False
from typing import List
from tqdm import tqdm


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
                             Y_dev: np.ndarray, noise=False, random_subset = 1.) -> np.ndarray:
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
    labels_set = list(set(Y_train.tolist()))

    if noise:
        print("Adding noise.")
        mean = np.mean(np.abs(X_train))
        mask_train = 0.0075 * (np.random.rand(*X_train.shape) - 0.5)

        X_train_cp += mask_train

    for i in tqdm(range(num_classifiers)):

        x_t, y_t = X_train_cp, Y_train

        clf = svm_classifier.SVMClassifier(classifier_class(**cls_params))

        idx = np.random.rand(x_y.shape[0]) < random_subset
        acc = clf.train_network(x_t[idx], y_t[idx], X_dev_cp, Y_dev)
        print("Iteration {}, Accuracy: {}".format(i, acc))
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
                                 X, Y, X, Y, noise=noise)

    print(list(zip(X.dot(P)[0], X.dot(P).dot(P)[0]))[:10])

    print(list(zip(P[0], P.dot(P)[0]))[:10])
    # assert np.allclose(P.dot(P), P)
    # assert np.allclose(P.dot(P), P.T)
    print("yay")
