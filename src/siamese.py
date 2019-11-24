import urllib
import spacy
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt

import random
import sklearn
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE

import torch
from torch import utils

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from torch.utils import data as D
from torch.nn import functional as F
from scipy import linalg
import scipy
from scipy.stats.stats import pearsonr
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




class Dataset(torch.utils.data.Dataset):
    
    """Simple torch dataset class"""
    def __init__(self, data, label):

        self.data = data
        self.label = label
        
        self.data_pairs = []
        self.labels_pairs = []
        
        for i in range(max(len(data), 6000)):
        
                j,k = np.random.choice(range(len(data)), size = 2)
                x,y = self.data[j], self.data[k]
                label = (self.label[j] == self.label[k]).astype(float)
                self.data_pairs.append((x,y))
                self.labels_pairs.append(label)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        with torch.no_grad():
             
            vec1, vec2 = self.data_pairs[index]
            label = self.labels_pairs[index]
            
            vec1, vec2, label = torch.from_numpy(vec1).float(), torch.from_numpy(vec2).float(), self.labels_pairs[index]
            return (vec1, vec2, label)


class Siamese(pl.LightningModule):

    def __init__(self, X_train, Y_train, X_dev, Y_dev):
        super(Siamese, self).__init__()
        self.l = torch.nn.Linear(300,1)
        self.train_data = Dataset(X_train, Y_train)
        self.dev_data = Dataset(X_dev, Y_dev)
        self.train_gen = torch.utils.data.DataLoader(self.train_data, batch_size = 8, drop_last = False, shuffle=True)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_data, batch_size = 8, drop_last = False, shuffle=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.binary_crossentropy = torch.nn.BCELoss()
        self.acc = None
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay = 1e-6)
        
    def forward(self, x1, x2):

          h1 = self.l(x1)
          h2 = self.l(x2)
          res = torch.diag(h1@(torch.t(h2)))
          return res
 
    def train_network(self, x_t, y_t, X_dev_cp, Y_dev):
    
      trainer = Trainer(max_nb_epochs = 14, min_nb_epochs = 8, show_progress_bar = False)
      trainer.fit(self)

      return self.acc   
      
    def get_weights(self):
    
        return self.l.weight.data.numpy()
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        return {'loss': self.binary_crossentropy(self.sigmoid(y_hat), y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x1, x2, y = batch
        y_hat = self.forward(x1, x2)
        y_argmax = y_hat > 0.5

        good = (y == y_argmax).sum().float()
        bad = (x1.shape[0] - good).float()
        acc = (good / (good + bad)).float()

        return {'val_loss': self.binary_crossentropy(self.sigmoid(y_hat), y), 'val_acc': acc}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.acc = avg_acc
        #print("Acc is {}".format(avg_acc))
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), weight_decay = 1e-4)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_gen

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        # can also return a list of val dataloaders
        return self.dev_gen
