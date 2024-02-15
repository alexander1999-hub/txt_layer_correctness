from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from typing import List

from abstract_feature_extractor import AbstractFeatureExtractor


class TFIDF(AbstractFeatureExtractor):
    def __init__(self, train_data: List[str]):
        self.vectorizer = Pipeline([
            ('vect', CountVectorizer(analyzer='char')),
            ('tfidf', TfidfTransformer()),
        ])
        self.vectorizer.fit(train_data)
        self.name = "TF-IDF"

    def transform(self, texts: List[str]) -> List:
        transformed_data = self.vectorizer.transform(texts)
        return transformed_data
