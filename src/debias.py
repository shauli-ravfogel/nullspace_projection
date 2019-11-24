import numpy as np
import scipy
from scipy import linalg
from typing import Tuple
import classifier
import svm_classifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from siamese import Siamese
REGRESSION = False

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
           min_accuracy: float, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray, noise = False, random_subset = True, regression = False, siamese = False) -> np.ndarray:

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


    for i in range(num_classifiers):

        if random_subset:
                if regression: raise Exception("subset not implemened for regression")
                
                #idx = np.random.rand(X_train_cp.shape[0]) < 0.5
                #x_t, y_t = X_train_cp[idx], Y_train[idx]
                class_to_decrease = np.random.choice(labels_set)
                idx_class =  Y_train == class_to_decrease
                idx_to_remove = np.random.rand(X_train_cp.shape[0]) > 0.25
                idx_to_maintain = idx_class*idx_to_remove + ~idx_class
                
                x_t, y_t = X_train_cp[idx_to_maintain], Y_train[idx_to_maintain]
        else:
        
                x_t,y_t = X_train_cp, Y_train
                
        #clf = classifier_class()
        if np.random.random() < 0.3:
                clf = svm_classifier.SVMClassifier(SGDClassifier(max_iter=7500, fit_intercept = True, penalty = "l2"))
                #clf = svm_classifier.SVMClassifier(LinearDiscriminantAnalysis(n_components = 1))
        else:
                clf = svm_classifier.SVMClassifier(LinearSVC(max_iter=35000, fit_intercept=True, class_weight="balanced", penalty="l2", dual=False))
        
        if regression:
                if np.random.random() < 0.5:
                        clf =  svm_classifier.SVMClassifier(SGDRegressor(average = True))
                else:
                        clf = svm_classifier.SVMClassifier(SVR(max_iter=35000, kernel = "linear"))
        
        if siamese and i > num_classifiers/2:
                if regression:
                        raise Exception("subset not implemened for regression") 
        
                clf = Siamese(x_t, y_t, X_dev_cp, Y_dev)
                

        acc = clf.train_network(x_t, y_t, X_dev_cp, Y_dev)
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
    
    
if __name__ == '__main__':
    X = np.random.rand(1000,300)
    Y = np.random.rand(1000) < 0.5
    num_classifiers = 3
    classifier_class = None
    input_dim = 300
    is_autoregressive = True
    min_accuracy = 0.5
    noise = False
    random_subset = True
    siamese = True
    
    P = get_debiasing_projection(classifier_class, num_classifiers, input_dim, is_autoregressive, min_accuracy, X,Y,X,Y, noise = noise, random_subset = random_subset, siamese = siamese)
    
    print(list(zip(X.dot(P)[0], X.dot(P).dot(P)[0]))[:10])
    
    print(list(zip(P[0], P.dot(P)[0]))[:10])   
    assert np.allclose(P.dot(P), P)
    assert np.allclose(P.dot(P), P.T)
    print("yay")
    
