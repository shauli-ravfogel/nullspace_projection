from typing import List, Tuple
import numpy as np
import copy


class DatasetHandler(object):

    def __init__(self, X_train: Tuple[np.ndarray], Y_train: np.ndarray, X_dev: Tuple[np.ndarray], Y_dev: np.ndarray,
                 dropout_rate = 0, Y_train_main = None, Y_dev_main = None, by_class = False, equal_chance_for_main_task_labels = True):
        self.X_train_original = X_train
        self.X_train_current = copy.deepcopy(X_train)
        self.Y_train = Y_train
        self.X_dev_original = X_dev
        self.X_dev_original = copy.deepcopy(X_dev)
        self.Y_dev = Y_dev
        self.dropout_rate = dropout_rate
        self.Y_train_main = Y_train_main
        self.Y_dev_main = Y_dev_main
        self.by_class = by_class
        self.equal_chance_for_main_task_labels = equal_chance_for_main_task_labels

        if by_class:
            if (Y_train_main is None) or (Y_dev_main is None):
                raise Exception("Need main-task labels for by-class training.")

    def apply_projection(self, P: np.ndarray):
        """
        apply the projection amtrix P on the current dataset
        :param P: a projection matrix
        """
        raise NotImplementedError

    def get_relevant_idx(self) -> Tuple[np.ndarray]:
        """
        this function filter only the relevant indices from the training & dev set.
        if by_class=False, all idx are relevant; othewise, randomly choose the indices corrsponding to a single
        main-task label, and keep only them for this iteration of INLP.
        :return: a tuple of numpy arrays, train_relevant_idx and dev_relevant_idx
        """
    def get_current_training_set(self) -> Tuple[np.ndarray]:

        raise NotImplementedError

    def get_current_dev_set(self) -> Tuple[np.ndarray]:

        raise NotImplementedError



class ClassificationDatasetHandler(DatasetHandler):

    def __init__(self, X_train: Tuple[np.ndarray], Y_train: np.ndarray, X_dev: Tuple[np.ndarray], Y_dev: np.ndarray,
                 dropout_rate = 0, Y_train_main = None, Y_dev_main = None, by_class = False, equal_chance_for_main_task_labels = True):

            super().__init__(X_train, Y_train, X_dev, Y_dev,
                 dropout_rate, Y_train_main, Y_dev_main, by_class, equal_chance_for_main_task_labels)

    def apply_projection(self, P):

        self.X_train_current = P.dot(self.X_train_original.T).T
        self.X_dev_current = P.dot(self.X_dev_original.T).T

    def get_relevant_idx(self) -> Tuple[np.ndarray]:

        if self.by_class:
            if self.equal_chance_for_main_task_labels:
                main_task_labels = list(set(self.Y_train_main.tolist()))
                cls = random.choice(main_task_labels)
            else:
                cls = np.random.choice(self.Y_train_main)
            relevant_idx_train = self.Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls

        else:

            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        return relevant_idx_train, relevant_idx_dev

    def get_current_training_set(self) -> Tuple[np.ndarray]:

        if self.dropout_rate > 0:
            dropout_scale = 1. / (1 - self.dropout_rate + 1e-6)
            dropout_mask = (np.random.rand(*self.X_train_current.shape) < (1 - dropout_rate)).astype(float) * dropout_scale

        relevant_idx_train , relevant_idx_dev = self.get_relevant_idx()

        return self.X_train_current * dropout_mask[relevant_idx_train], self.Y_train[relevant_idx_train]

    def get_current_dev_set(self) -> Tuple[np.ndarray]:

        relevant_idx_train, relevant_idx_dev = self.get_relevant_idx()
        return self.X_dev_current[relevant_idx_dev], self.Y_dev[relevant_idx_dev]

