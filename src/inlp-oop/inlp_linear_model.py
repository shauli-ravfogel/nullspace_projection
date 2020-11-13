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
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
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


class TorchLinearModel(pl.LightningModule):

    def __init__(self, dim, device, use_bias = True, num_epochs=10):
        super().__init__()
        
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = torch.nn.Linear(dim,1, bias = use_bias)
        self.device = device
        self.dim = dim
        self.num_epochs = num_epochs
        
    """implementing the classifier interface"""
    
    def initialize_model(self):

        self.model = torch.nn.Linear(self.dim, 1)
        
    def train_model(self, dataset_handler: inlp_dataset_handler.ClassificationDatasetHandler) -> float:

        """
        :param dataset_handler:
        :return:  accuracy score on the dev set / Person's R in the case of regression
        """

        X_train, Y_train = dataset_handler.get_current_training_set()
        X_dev, Y_dev = dataset_handler.get_current_dev_set()
        train_loader = DataLoader(list(zip(X_train, Y_train)), batch_size = 16)
        dev_loader = DataLoader(list(zip(X_dev, Y_dev)), batch_size = 16)
        trainer = pl.Trainer(max_nb_epochs=self.num_epochs, min_nb_epochs=1, gpus = 1 if self.device == "cuda" else 0)
        trainer.fit(self, train_loader, dev_loader)

        score = self.score(X_dev, Y_dev)
        return score
    
    def score(self, X_dev, Y_dev):
    
        dev_loader = DataLoader(list(zip(X_dev, Y_dev)), batch_size = 16)
        accs = []
        
        for x,y in dev_loader:
        
            y_hat = self(x)
            acc = self.calc_accuracy(y_hat, y)
            accs.append(acc.detach().cpu().numpy().item())
        
        return np.mean(accs)
        
        
    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.weight.detach().cpu().numpy()
        
        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)

        return w  
             
    """ pytorch-lightning functions"""
    
    def forward(self, x):
        return self.model(x.float()).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.bce_loss_fn(y_hat, y.float())
        return {"loss": loss}
    
    def calc_accuracy(self, y_hat, y):
    
        return (torch.abs(torch.sigmoid(y_hat)-y) < 0.5).sum()/(1.*len(y_hat))
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.bce_loss_fn(y_hat, y.float())
        acc = self.calc_accuracy(y_hat, y)

        return {"loss": loss, "acc": acc}
        #self.log('valid_loss', loss)
    
    def validation_end(self, outputs):
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        print("ACC: {}".format(avg_acc.detach().cpu().numpy().item()))
        return {"avg_acc": avg_acc}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
        #return torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.75)
        
                
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
            return vec1, vec2, torch.tensor(y).to(self.device).float()


class MetricLearningDataset(torch.utils.data.Dataset):
    """Simple torch dataset class"""

    def __init__(self, x1: np.ndarray, x2: np.ndarray, sents1: np.ndarray, sents2: np.ndarray, ids1: np.ndarray, ids2: np.ndarray, device):

        self.x1 = x1
        self.x2 = x2
        self.sents1 = sents1
        self.sents2 = sents2
        self.ids1 = ids1
        self.ids2 = ids2 

        self.device = device

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        with torch.no_grad():
            v1, v2= self.x1[index], self.x2[index]

            vec1, vec2 = torch.from_numpy(v1).double(), torch.from_numpy(v2).double()
            vec1 = vec1.to(self.device)
            vec2 = vec2.to(self.device)
            
            return vec1, vec2, self.sents1[index], self.sents2[index], self.ids1[index], self.ids2[index]


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




class SiameseMetricLearning(LinearModel):

    def __init__(self, model_class = siamese_model.Siamese, model_params = {}):
        """

        :param model_class: the class of the siamese model (default - pytorch-lightning implementation)
        :param model_params: a dict, specifying model_class initialization parameters
        :param concat_weights: bool. If true, concat the siamese weights; otherwise, average them.
                NOTE: if False, the nullspace projection matrix is not guaranteed to project to the nullspace of l1, l2
                NOTE: this distinction is only meaningful when the siamese network uses different weight matrices for the two inputs.
        """
        self.model_class = model_class
        self.model_params = model_params
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

        X_train, sents_train, ids_train = dataset_handler.get_current_training_set()
        X_train1, X_train2 = X_train
        X_dev, sents_dev, ids_dev = dataset_handler.get_current_dev_set()
        X_dev1, X_dev2 = X_dev

        device = self.model_params["device"]
        ids_train1, ids_train2 = ids_train
        ids_dev1, ids_dev2 = ids_dev
        sents_train1, sents_train2 = sents_train
        sents_dev1, sents_dev2 = sents_dev
        
        train_dataset = MetricLearningDataset(X_train1, X_train2, sents_train1, sents_train2, ids_train1, ids_train2, device = device)
        dev_dataset = MetricLearningDataset(X_dev1, X_dev2, sents_dev1, sents_dev2, ids_dev1, ids_dev2, device = device)

        self.model = self.model_class(train_dataset, dev_dataset, input_dim = self.model_params["input_dim"], hidden_dim = self.model_params["hidden_dim"], batch_size = self.model_params["batch_size"], verbose = self.model_params["verbose"], k = self.model_params["k"], p = self.model_params["p"], mode = self.model_params["mode"], final = self.model_params["final"], device = self.model_params["device"])
        self.model = self.model.to(device)
        score = self.model.train_network(self.model_params["num_iter"])
        return score

    def get_weights(self) -> np.ndarray:
        """
        :return: final weights of the model, as np array
        """

        w = self.model.l.weight.detach().cpu().numpy()

        if len(w.shape) == 1:
                w = np.expand_dims(w, 0)
        return w
