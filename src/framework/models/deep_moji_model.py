from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import BooleanAccuracy, F1Measure
from overrides import overrides


@Model.register("model_deep_moji")
class DeepMojiModel(Model):
    """
    simple MLP model

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    emb_size: ``int``, required
        the size of the input embedding
    hidden_size: ``int``, required
        the size of the hidden layer
    """

    def __init__(self,
                 vocab: Vocabulary,
                 emb_size: int,
                 hidden_size: int,
                 ) -> None:
        super().__init__(vocab)
        self.emb_size = emb_size

        # an mlp with one hidden layer
        layers = []
        layers.append(nn.Linear(self.emb_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.scorer = nn.Sequential(*layers)

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}

    @overrides
    def forward(self,
                vec: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        vec: torch.Tensor, required
            The input vector.
        label: torch.Tensor, optional (default = None)
            A variable of the correct label.

        Returns
        -------
        An output dictionary consisting of:
        y_hat: torch.FloatTensor
            the predicted values
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        scores = self.scorer(vec)
        y_hat = torch.argmax(scores, dim=1)

        output = {"y_hat": y_hat}
        if label is not None:
            self.metrics['accuracy'](y_hat, label)
            self.metrics['f1'](torch.nn.functional.softmax(scores, dim=1), label)
            output["loss"] = self.loss(scores, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self.metrics['f1'].get_metric(reset=reset)
        return {"accuracy": self.metrics['accuracy'].get_metric(reset=reset),
                "p": p,
                "r": r,
                "f1": f1}
