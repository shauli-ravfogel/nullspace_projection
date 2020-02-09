from typing import List, Tuple
import numpy as np
import copy


class DatasetHandler(object):

    def __init__(self, X_train: Tuple[np.ndarray], Y_train: np.ndarray, X_dev: Tuple[np.ndarray], Y_dev: np.ndarray,
                 dropout_rate = 0, X_main_task = None, Y_main_task = None):
        self.X_train_original = X_train
        self.X_train_current = copy.deepcopy(X_train)
        self.Y_train = Y_train
        self.X_dev_original = X_dev
        self.X_dev_original = copy.deepcopy(X_dev)
        self.Y_dev = Y_dev
        self.dropout_rate = dropout_rate
        self.X_main_task = X_main_task
        self.Y_main_task = Y_main_task

    def apply_projection(self, P: np.ndarray):
        """
        apply the projection amtrix P on the current dataset
        :param P: a projection matrix
        """
        raise NotImplementedError

    def get_current_training_set(self) -> Tuple[np.ndarray]:

        raise NotImplementedError

    def get_current_dev_set(self) -> Tuple[np.ndarray]:

        raise NotImplementedError



class ClassificationDatasetHandler(DatasetHandler):

    def __init__(self, X_train: Tuple[np.ndarray], Y_train: np.ndarray, X_dev: Tuple[np.ndarray], Y_dev: np.ndarray,
                 dropout_rate = 0, X_main_task = None, Y_main_task = None):

            super().__init__(X_train, Y_train, X_dev, Y_dev,
                 dropout_rate, X_main_task, Y_main_task)

    def apply_projection(self, P):

        self.X_train_current = P.dot(self.X_train_original.T).T
        self.X_dev_current = P.dot(self.X_dev_original.T).T

    def get_current_training_set(self) -> Tuple[np.ndarray]:

        return self.X_train_current, self.Y_train

    def get_current_dev_set(self) -> Tuple[np.ndarray]:

        return self.X_dev_current, self.Y_dev

