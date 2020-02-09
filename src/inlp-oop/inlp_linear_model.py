import numpy as np
import inlp_dataset_handler
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

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

    def __init__(self, model_class: Siamese, model_params):
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
        X_train1, X_train2 = X_train
        X_dev, Y_dev = dataset_handler.get_current_dev_set()
        X_dev1, X_dev2 = X_dev

        train_dataset = Dataset(X_train1, X_train2, Y_train, device = "cpu")
        dev_dataset = Dataset(X_dev1, X_dev2, Y_dev, device="cpu")

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


class Siamese(pl.LightningModule):

    def __init__(self, train_dataset: torch.utils.Dataset, dev_dataset: torch.utils.Dataset, dim, batch_size, device="cuda"):

        super(Siamese, self).__init__()
        d = 32
        self.l1 = torch.nn.Linear(dim, 32, bias=True)
        self.l2 = torch.nn.Linear(dim, 32, bias=True)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.w1, self.w2, self.w3, self.b = torch.nn.Parameter(torch.rand(1)), torch.nn.Parameter(
            torch.rand(1)), torch.nn.Parameter(torch.rand(1)), torch.nn.Parameter(torch.zeros(1))

        self.train_data = Dataset(X_train, device)
        self.dev_data = Dataset(X_dev, device)
        self.train_gen = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, drop_last=False,
                                                     shuffle=True)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_data, batch_size=batch_size, drop_last=False, shuffle=True)
        self.acc = None
        # self.optimizer = torch.optim.Adam(self.parameters(), weight_decay = 1e-6)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, 16))

    def forward(self, x1, x2):
        h1 = self.l1(x1)
        h2 = self.l2(x2)

        # h1, h2 = self.net(x1), self.net(x2)

        return h1, h2

    def train_network(self, num_epochs):
        trainer = Trainer(max_nb_epochs=num_epochs, min_nb_epochs=num_epochs, show_progress_bar=False)
        trainer.fit(self)

        return self.acc

    def get_weights(self):
        return self.l.weight.data.detach().cpu().numpy()

    def get_final_representaton_for_sigmoid(self, h1, h2):
        # norm1,norm2 =  torch.norm(h1, dim = 1, keepdim = True),  torch.norm(h2, dim = 1, keepdim = True)
        # dot_prod = torch.sum(h1 * h2, axis = 1)
        dot_prod = self.cosine_sim(h1, h2)
        # dot_prod = torch.sum((h1-h2)**2, axis = 1)
        # dot_prod = torch.sum(h1 - h2, axis = 1)
        dot_prod = self.w3 * dot_prod + self.b
        return dot_prod

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x1, x2, y = batch
        h1, h2 = self.forward(self.dropout(x1), self.dropout(x2))
        dot_prod = self.get_final_representaton_for_sigmoid(h1, h2)

        loss_val = self.loss_fn(dot_prod, y)
        correct = ((dot_prod > 0).int() == y.int()).int()
        acc = torch.sum(correct).float() / len(y)

        return {'loss': loss_val, 'val_acc': acc}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x1, x2, y = batch
        h1, h2 = self.forward(x1, x2)
        dot_prod = self.get_final_representaton_for_sigmoid(h1, h2)

        loss_val = self.loss_fn(dot_prod, y)
        correct = ((dot_prod > 0).int() == y.int()).int()
        acc = torch.sum(correct).float() / len(y)

        return {'val_loss': loss_val, 'val_acc': acc}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print("Loss is {}".format(avg_loss))
        print("Accuracy is {}".format(avg_acc))
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # return torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        return torch.optim.Adam(self.parameters(), weight_decay=1e-4)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_gen

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return self.dev_gen