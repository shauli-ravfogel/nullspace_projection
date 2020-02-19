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

            vec1, vec2 = torch.from_numpy(v1).double(), torch.from_numpy(v2).double()
            vec1 = vec1.to(self.device)
            vec2 = vec2.to(self.device)

            return vec1, vec2, torch.tensor(y).float().to(self.device)



class SiameseLinearClassifier(LinearModel):

    def __init__(self, model_class = siamese_model.Siamese, model_params = {}, concat_weights = True):
        """

        :param model_class: the class of the siamese model (default - pytorch-lightning implementation)
        :param model_params: a dict, specifying model_class initialization parameters
        :param concat_weights: bool. If true, concat the siamese weights; otherwise, average them.
                NOTE: if False, the nullspace projection matrix is not guaranteed to project to the nullspace of l1, l2
                NOTE: this distinction is only meaningful when the siamese network uses different weight matrices for the two inputs.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.concat_weights = concat_weights
        self.initialize_model()

    def initialize_model(self):
        """
        not in use (model initialziation needs a dataset in torch-lightning)
        :return:
        """
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

        device = self.model_params["device"]
        train_dataset = Dataset(X_train1, X_train2, Y_train, device = device)
        dev_dataset = Dataset(X_dev1, X_dev2, Y_dev, device = device)

        self.model = self.model_class(train_dataset, dev_dataset, input_dim = self.model_params["input_dim"], hidden_dim = self.model_params["hidden_dim"], batch_size = self.model_params["batch_size"], verbose = self.model_params["verbose"], same_weights = self.model_params["same_weights"], compare_by = self.model_params["compare_by"]).to(device)
        score = self.model.train_network(self.model_params["num_iter"])
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w1, w2 = self.model.l1.weight.detach().cpu().numpy(), self.model.l2.weight.detach().cpu().numpy()

        if self.concat_weights:
            w = np.concatenate([w1, w2], axis = 0)
        else:
            w = (w1 + w2) / 2

        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)
        return w



