import numpy as np
import inlp_dataset_handler

# an abstract class for linear classifiers

class LinearModel(object):

    def __init__(self):

        pass

    def train_model(self, dataset_handler: inlp_dataset_handler.DatasetHandler) -> float:
        """

        :param dataset_handler: an instance of DatasetHandler
        :return: accuracy score on the dev set
        """

        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        raise NotImplementedError




class SKlearnClassifier(LinearModel):

    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.initialize_model()

    def initialize_model(self):

        model = self.model_class(**self.model_params)
        self.model = model

    def train_train_model(self, dataset_handler: inlp_dataset_handler.ClassificationDatasetHandler) -> float:

        """
        :param dataset_handler:
        :return:  accuracy score on the dev set / Person's R in the case of regression
        """

        X_train, Y_train = dataset_handler.get_current_training_set()
        X_dev, Y_dev = dataset_handler.get_current_dev_set()

        self.model.fit(X_train, Y_train)
        score = self.model.score(X_dev, Y_dev)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.coef_
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w
