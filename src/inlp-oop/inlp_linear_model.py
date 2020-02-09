import numpy as np
import inlp_dataset_handler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import siamese_model
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

    def train_model(self, dataset_handler: inlp_dataset_handler.ClassificationDatasetHandler) -> float:

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


class Dataset(torch.utils.data.Dataset):
    """Simple torch dataset class"""

    def __init__(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray, device):

        self.x1 = x1
        self.x2 = x2
        self.y = y

        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        with torch.no_grad():
            v1, v2, y = self.x1[index], self.x2[index], self.y[index]

            vec1, vec2 = torch.from_numpy(v1).float(), torch.from_numpy(v2).float()
            vec1 = vec1.to(self.device)
            vec2 = vec2.to(self.device)

            return vec1, vec2, torch.tensor(y).float().to(self.device)



class SiameseLinearClassifier(LinearModel):

    def __init__(self, model_class: siamese_model.Siamese, model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.initialize_model()

    def initialize_model(self):
        return

        model = self.model_class(**self.model_params)
        self.model = model

    def train_model(self, dataset_handler: inlp_dataset_handler.ClassificationDatasetHandler) -> float:

        """
        :param dataset_handler:
        :return:  accuracy score on the dev set / Person's R in the case of regression
        """

        X_train, Y_train = dataset_handler.get_current_training_set()
        X_train1, X_train2 = X_train
        X_dev, Y_dev = dataset_handler.get_current_dev_set()
        X_dev1, X_dev2 = X_dev

        train_dataset = Dataset(X_train1, X_train2, Y_train, device = "cpu")
        dev_dataset = Dataset(X_dev1, X_dev2, Y_dev, device =" cpu")

        self.model = self.model_class(train_dataset, dev_dataset, dim = 32, batch_size = 32, device = "cpu")
        score = self.model.train_network(15)
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.l1.weight.detach().cpu().numpy() + self.model.l2.weight.detach().cpu().numpy()
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w



