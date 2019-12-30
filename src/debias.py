
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
    print(nullspace_basis)
    exit()
    nullspace_basis = nullspace_basis * np.sign(nullspace_basis[0][0])  # handle sign ambiguity
    projection_matrix = nullspace_basis.dot(nullspace_basis.T)

    return projection_matrix
    
    
def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix
    """

    w_basis = scipy.linalg.orth(W.T) # orthogonal basis
    w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
    
    return P_W



def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):

    rowspace_projections = []
    
    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    Q = np.sum(rowspace_projections, axis = 0)
    P = I - get_rowspace_projection(Q)
    
    return P


def get_debiasing_projection(classifier_class, cls_params: Dict, num_classifiers: int, input_dim: int,
                             is_autoregressive: bool,
                             min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray,
                             Y_dev: np.ndarray, by_class=True, Y_train_main=None,
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
    :param by_class: if true, at each iteration sample one main-task label, and extract the protected attribute only from vectors from this class
    :param T_train_main: ndarray, main-task train labels
    :param Y_dev_main: ndarray, main-task eval labels
    :return: P, the debiasing projection; rowspace_projections, the list of all rowspace projection; Ws, the list of all calssifiers.
    """
    
    I = np.eye(input_dim)
    
    if by_class:
        if ((Y_train_main is None) or (Y_dev_main is None)):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))
                            
    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []
    
    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        
        clf = svm_classifier.SVMClassifier(classifier_class(**cls_params))

        if by_class:
            cls = np.random.choice(Y_train_main)  # random.choice(main_task_labels) UNCOMMENT FOR EQUAL CHANCE FOR ALL Y
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network(X_train_cp[relevant_idx_train], Y_train[relevant_idx_train], X_dev_cp[relevant_idx_dev], Y_dev[relevant_idx_dev])
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy: continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            
            """
            to ensure numerical stability, explicitly project to the intersection of the nullspaces found so far (instaed of doing X = P_iX,
            which is problematic when w_i is not exactly orthogonal to w_i-1,...,w1, due to e.g inexact argmin calculation).
            """
            
            # use the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf: 
            # N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
            
            Q = np.sum(rowspace_projections, axis = 0)
            P = I - get_rowspace_projection(Q)
                 
            # project  
                      
            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    """
    calculae final projection matrix P=PnPn-1....P2P1
    since w_i.dot(w_i-1) = 0, P2P1 = I - P1 - P2 (proof in the paper); this is more stable.
    by induction, PnPn-1....P2P1 = I - (P1+..+PN). We will use instead Ben-Israel's formula to increase stability,
    i.e., we explicitly project to intersection of all nullspaces (not very critical)
    """
    
    Q = np.sum(rowspace_projections, axis = 0)
    P = I - get_rowspace_projection(Q)
    P_alternative = I - np.sum(rowspace_projections, axis = 0)
    #print(np.allclose(P,P_alternative))
    
    return P, rowspace_projections, Ws


if __name__ == '__main__':
    N = 10000
    d = 5
    X = np.random.rand(N, d) - 0.5
    Y = np.array([1 if sum(x) > 0 else 0 for x in X]) #X < 0 #np.random.rand(N) < 0.5 #(X + 0.01 * (np.random.rand(*X.shape) - 0.5)) < 0 #np.random.rand(5000) < 0.5
    #Y = np.array(Y, dtype = int)
    
    num_classifiers = 3
    classifier_class = SGDClassifier #Perceptron
    input_dim = d
    is_autoregressive = True
    min_accuracy = 0.0
    noise = False
    random_subset = True
    
    P = get_debiasing_projection(classifier_class, {"max_iter": 15}, num_classifiers, input_dim, is_autoregressive, min_accuracy, X, Y, None, None, by_class = False)
    
