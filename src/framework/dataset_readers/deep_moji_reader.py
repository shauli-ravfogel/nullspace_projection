import logging
from typing import Dict

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ArrayField
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("deep_moji_reader")
class DeepMojiReader(DatasetReader):
    """

    """

    def __init__(self,
                 ratio: float = 0.5,
                 n: int = 100000,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.n = n
        self.ratio = ratio

    @overrides
    def _read(self, dir_path: str):
        logger.info("Reading instances from lines in file at: %s", dir_path)

        n_1 = int(self.n * self.ratio / 2)
        n_2 = int(self.n * (1 - self.ratio) / 2)

        for file, label, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                        ['positive', 'positive', 'negative', 'negative'],
                                        [n_1, n_2, n_2, n_1]):
            data = np.load('{}/{}.npy'.format(dir_path, file))

            for vec in data[:class_n]:
                yield self.text_to_instance(vec, label)

    @overrides
    def text_to_instance(self, vec: np.ndarray,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {'vec': ArrayField(vec)}

        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
