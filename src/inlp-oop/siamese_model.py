import numpy as np
import inlp_dataset_handler
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Siamese(pl.LightningModule):

    def __init__(self, train_dataset: Dataset, dev_dataset: Dataset, input_dim, hidden_dim, batch_size,
                 verbose=True, same_weights=True, compare_by: str = "cosine"):
        """
        :param train_dataset: a pytorch Dataset instance
        :param dev_dataset: a pytorch Dataset instance
        :param input_dim: dimensionality of input vectors
        :param hidden_dim: dimensionality of hidden representations
        :param batch_size:
        :param verbose: bool. If true, print training progress
        :param same_weights: bool. If true, the model uses the same weight matrix for both inputs
        :param compare_by: str, the way to derive the sigmoid score from the two representations.
                options: 'cosine' (cosine distance); "l2" (euclidean distance); 'dot_product'
        """
        super().__init__()

        self.l1 = torch.nn.Linear(input_dim, hidden_dim, bias=True).double()
        if not same_weights:
                self.l2 = torch.nn.Linear(input_dim, hidden_dim, bias=True).double()
        else:
                self.l2 = self.l1.double()

        self.same_weights = same_weights
        self.verbose = verbose
        self.compare_by = compare_by
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.w_out, self.b_out = torch.nn.Parameter(torch.rand(1)), torch.nn.Parameter(torch.zeros(1))

        self.train_data = train_dataset
        self.dev_data = dev_dataset
        self.train_gen = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, drop_last=False,
                                                     shuffle=True)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_data, batch_size=batch_size, drop_last=False, shuffle=True)
        self.acc = None
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x1, x2):

        h1 = self.l1(x1)
        h2 = self.l2(x2)
        return h1, h2

    def train_network(self, num_epochs):
        trainer = Trainer(max_nb_epochs=num_epochs, min_nb_epochs=num_epochs, show_progress_bar=self.verbose)
        trainer.fit(self)
        return self.acc

    def get_final_representaton_for_sigmoid(self, h1, h2):

        """
        :param h1:
        :param h2:
        :return: the similarity score of h1 and h2
        """

        if self.compare_by == "cosine":
            scores = self.cosine_sim(h1, h2)
        elif self.compare_by == "dot_product":
            scores = torch.sum(h1 * h2, axis = 1)
        elif self.compare_by == "l2":
            scores = torch.sum((h1 - h2) ** 2, axis = 1)
        else:
            raise Exception("Unsupported comparison method")

        scores = self.w_out * scores + self.b_out
        return scores

    def training_step(self, batch, batch_nb):
        x1, x2, y = batch
        h1, h2 = self.forward(x1, x2)
        similarity_scores = self.get_final_representaton_for_sigmoid(h1, h2)

        loss_val = self.loss_fn(similarity_scores, y)
        correct = ((similarity_scores > 0).int() == y.int()).int()
        acc = torch.sum(correct).float() / len(y)

        return {'loss': loss_val, 'val_acc': acc}

    def validation_step(self, batch, batch_nb):
        x1, x2, y = batch
        h1, h2 = self.forward(x1, x2)
        similarity_scores = self.get_final_representaton_for_sigmoid(h1, h2)

        loss_val = self.loss_fn(similarity_scores, y)
        correct = ((similarity_scores > 0).int() == y.int()).int()
        acc = torch.sum(correct).float() / len(y)

        return {'val_loss': loss_val, 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # print("Loss is {}".format(avg_loss))
        # print("Accuracy is {}".format(avg_acc))
        self.acc = avg_acc
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
