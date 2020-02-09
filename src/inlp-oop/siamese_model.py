import numpy as np
import inlp_dataset_handler
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Siamese(pl.LightningModule):

    def __init__(self, train_dataset: Dataset, dev_dataset: Dataset, dim, batch_size, device="cuda"):

        super().__init__()
        d = 32
        self.l1 = torch.nn.Linear(dim, d, bias=True)
        self.l2 = torch.nn.Linear(dim, d, bias=True)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
        self.w1, self.w2, self.w3, self.b = torch.nn.Parameter(torch.rand(1)), torch.nn.Parameter(
            torch.rand(1)), torch.nn.Parameter(torch.rand(1)), torch.nn.Parameter(torch.zeros(1))

        self.train_data = train_dataset
        self.dev_data = dev_dataset
        self.train_gen = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, drop_last=False,
                                                     shuffle=True)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_data, batch_size=batch_size, drop_last=False, shuffle=True)
        self.acc = None
        # self.optimizer = torch.optim.Adam(self.parameters(), weight_decay = 1e-6)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x1, x2):
        h1 = self.l1(x1)
        h2 = self.l2(x2)
        return h1, h2

    def train_network(self, num_epochs):
        trainer = Trainer(max_nb_epochs=num_epochs, min_nb_epochs=num_epochs, show_progress_bar=True)
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
        h1, h2 = self.forward(x1, x2)
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
        #print("Loss is {}".format(avg_loss))
        #print("Accuracy is {}".format(avg_acc))
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
