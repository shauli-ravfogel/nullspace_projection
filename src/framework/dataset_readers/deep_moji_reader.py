import logging
from typing import Dict

import logging
from typing import Dict

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("deep_moji_reader")
class DeepMojiReader(DatasetReader):
    """

    """

    def __init__(self,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._pos_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, dir_path: str):
        # with open(cached_path(dir_path), 'r') as f:
        logger.info("Reading instances from lines in file at: %s", dir_path)
        for file in ['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg']:
            data = np.load('{}/{}.npy'.format(dir_path, file))

            for vec in data:
                if file.split('_')[0] == 'pos':
                    label = 'positive'
                else:
                    label = 'negative'
                yield self.text_to_instance(vec, label)

    @overrides
    def text_to_instance(self, vec: np.ndarray,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        fields['vec'] = ArrayField(vec)

        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
