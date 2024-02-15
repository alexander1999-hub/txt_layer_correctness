import joblib
import numpy as np
import os
import time
from sklearn.metrics import classification_report
from typing import Dict, List, Optional
from xgboost import XGBClassifier

from abstract_feature_extractor import AbstractFeatureExtractor
from abstract_model import AbstractModel


class XGBoost(AbstractModel):
    def __init__(self, from_checkpoint: bool, checkpoint_path: Optional[str] = ''):
        self.model = XGBClassifier(random_state=42, learning_rate=0.5, n_estimators=600, booster="gbtree", tree_method="hist",
                                   max_depth=3)
        self.__load_model(from_checkpoint, checkpoint_path)
        self.name = "XGBoost"

    def __load_model(self, from_checkpoint: bool, checkpoint_path: str):
        if from_checkpoint and os.path.isfile(checkpoint_path):
            self.model = joblib.load(checkpoint_path)

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int], parameters: Dict,
              feature_extractor: AbstractFeatureExtractor):

        self.model.fit(
            X=feature_extractor.transform(train_texts),
            y=train_labels,
            eval_set=[(feature_extractor.transform(eval_texts), eval_labels)],
            verbose_eval=False,
        )
        joblib.dump(self.model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'xgb.pkl'), compress=False)

        eval_pred = self.predict(eval_texts, parameters={}, feature_extractor=feature_extractor)
        print("Evaluation results")
        print(classification_report(eval_labels, eval_pred, digits=3))

    def test(self, test_texts: List[str], test_labels: List[int], parameters: Dict, feature_extractor:
             AbstractFeatureExtractor, measure_time: Optional[bool] = False):
        test_pred = []
        timings = []
        for text in test_texts:
            start = time.time()
            text_pred = self.predict([text], parameters={}, feature_extractor=feature_extractor)
            end = time.time()
            timings.append(end - start)
            test_pred.extend(text_pred)
        print(classification_report(test_labels, test_pred, digits=3))
        if measure_time:
            print("Average predict time:", np.mean(timings), "seconds")

    def fill_text_layer(self, text):
        if len(text) < 150:
            return f"\n{text}" * (150 // len(text))
        return text

    def predict(self, text_layer: List[str], parameters: Dict, feature_extractor: AbstractFeatureExtractor) -> List[int]:
        text_layer_fixed = [self.fill_text_layer(text) for text in text_layer]
        features = feature_extractor.transform(text_layer_fixed)
        prediction = self.model.predict(features)
        return prediction
