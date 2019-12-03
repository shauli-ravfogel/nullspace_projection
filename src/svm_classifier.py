from src import classifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
import numpy as np


class SVMClassifier(classifier.Classifier):

    def __init__(self, m):

        #self.model = LinearSVC(max_iter=50000, fit_intercept=True, class_weight="balanced", penalty="l2", dual=False)
        self.model = m #SGDClassifier(loss = "perceptron", penalty = "l2")

    def train_network(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

        """
        :param X_train:
        :param Y_train:
        :param X_dev:
        :param Y_dev:
        :return: accuracy score on the dev set
        """

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w  = np.expand_dims(w, 0)

        return w
