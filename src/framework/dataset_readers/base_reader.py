import json
import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("classification_reader")
class ClassificationReader(DatasetReader):
    """

    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._pos_indexers = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for ind, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                curr_example_json = json.loads(line)
                text = curr_example_json['text']

                label = curr_example_json['main_label']
                yield self.text_to_instance(text,
                                            label)
                if ind > 100:
                    break

    @overrides
    def text_to_instance(self, text: str,
                         label: int = None) -> Instance:

        fields: Dict[str, Field] = {}

        sentence = TextField([Token(t) for t in text], self._token_indexers)
        fields['text'] = sentence

        meta = {'text': text}
        fields['metadata'] = MetadataField(meta)

        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)
