import abc
from typing import List


class AbstractFeatureExtractor:
    @abc.abstractmethod
    def transform(self, texts: List[str]):
        pass
