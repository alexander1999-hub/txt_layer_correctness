import abc
from typing import Dict, List, Optional

from abstract_feature_extractor import AbstractFeatureExtractor


class AbstractModel:
    @abc.abstractmethod
    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int], parameters: Dict,
              feature_extractor: AbstractFeatureExtractor):
        pass

    @abc.abstractmethod
    def test(self, test_texts: List[str], test_labels: List[int], parameters: Dict, feature_extractor: AbstractFeatureExtractor,
             measure_time: Optional[bool] = False):
        pass

    @abc.abstractmethod
    def predict(self, text_layer: str, parameters: Dict, feature_extractor: AbstractFeatureExtractor):
        pass
