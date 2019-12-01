from typing import Dict, List, Any

from overrides import overrides

import torch
import numpy as np

from allennlp.common.checks import check_dimensions_match
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy, F1Measure


@Model.register("model_base")
class BaseModel(Model):
    """
    This ``Model`` make a classification of the FH problem.
    Given a sentence and an anchor (number) it create contextualized
    representation for every token and combined with the anchor it
    assign a score, as well to the 6 implicit classes.
    Both the scores for the implicit classes and the tokens are concatenated
    and the highest score is the models' prediction.


    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder: ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    mlp_dropout: ``float``, required (default = 0.2)
        The dropout probability of the mlp scorer.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 emb_size: int,
                 mlp_dropout: float = 0.2) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.emb_size = emb_size

        self.scorer = FeedForward(self.emb_size, num_layers=1,
                                  hidden_dims=[2], activations=[
                                    Activation.by_name('linear')()],
                                  dropout=mlp_dropout)

        self.loss = torch.nn.CrossEntropyLoss()
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'f1': F1Measure(positive_label=1)}

    @overrides
    def forward(self,
                text: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text: Dict[str, torch.Tensor], required
            The input sentence.
        metadata: List[Dict[str, Any]], required
        label: torch.Tensor, optional (default = None)
            A variable representing the index of the correct label.

        Returns
        -------
        An output dictionary consisting of:
        tag_logits: torch.FloatTensor, required
            A tensor of shape ``(batch_size, max_sentence_length)``
            representing a distribution over the label classes for each instance.
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embeddings = self.text_field_embedder(text)

        cls = embeddings[:, 0, :]

        scores = self.scorer(cls)
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
