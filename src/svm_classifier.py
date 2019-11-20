import classifier
from sklearn.svm import LinearSVC, SVC
import numpy as np


class SVMClassifier(classifier.Classifier):

    def __init__(self):

        self.model = LinearSVC(verbose=2,
                  max_iter=50000, fit_intercept=True, class_weight="balanced", penalty="l2", dual=False)


    def train(self, X_train: np.ndarray, Y_train: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:

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

        return self.model.coef_