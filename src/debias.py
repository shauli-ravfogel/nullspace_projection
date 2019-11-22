import numpy as np
import scipy
from scipy import linalg
from typing import Tuple
import classifier
import svm_classifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
           min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray, noise = False, random_projection = False) -> np.ndarray:

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
    
    if noise:
        print("Adding noise.")
        mean = np.mean(np.abs(X_train))
        mask_train = 0.01 * (np.random.rand(*X_train.shape) - 0.5)
        
        X_train_cp += mask_train

    for i in range(num_classifiers):

        #clf = classifier_class()
        if np.random.random() < 0.2:
                clf = svm_classifier.SVMClassifier(SGDClassifier(max_iter=7500, loss = "perceptron", penalty = "l2"))
                #clf = svm_classifier.SVMClassifier(LinearDiscriminantAnalysis(n_components = 1))
        else:
                clf = svm_classifier.SVMClassifier(LinearSVC(max_iter=25000, fit_intercept=False, class_weight="balanced", penalty="l2", dual=False))
                
        acc = clf.train(X_train_cp, Y_train, X_dev_cp, Y_dev)
        print("Iteration {}, Accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        P_i = get_nullspace_projection(W)
        #P = P_i.dot(P)
        P = P.dot(P_i)

        if is_autoregressive:

            X_train_cp = X_train_cp.dot(P_i)
            X_dev_cp = X_dev_cp.dot(P_i)

    return P
