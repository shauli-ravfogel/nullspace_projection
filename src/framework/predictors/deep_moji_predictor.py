from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register('deep_moji_predictor')
class DeepMojiPredictor(Predictor):
    """"Predictor wrapper for the base model"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict['text']
        label = json_dict['main_label'] if 'main_label' in json_dict else None
        instance = self._dataset_reader.text_to_instance(text=text,
                                                         label=label)

        return instance
