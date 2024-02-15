import os
import pickle
import re
import numpy as np

from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import ngrams


from abstract_model import AbstractModel
from abstract_feature_extractor import AbstractFeatureExtractor
from typing import List, Dict, Optional
import joblib
import time
import os
from sklearn.metrics import classification_report, f1_score


class NGram(AbstractModel):
    def __init__(self, from_checkpoint: bool, ngram_num: int, laplace: bool, checkpoint_path: Optional[str] = ''):
        self.ngram_num = ngram_num
        self.laplace = laplace
        self.name = str(self.ngram_num)+"-gram_"+['MLE', 'Laplace'][self.laplace]
        if self.laplace:
            self.model = Laplace(self.ngram_num)
        else:
            self.model = MLE(self.ngram_num)
        self.__load_model(from_checkpoint, checkpoint_path)


    def __load_model(self, from_checkpoint: bool, checkpoint_path: str):
        if from_checkpoint and os.path.isfile(checkpoint_path):
            self.model = joblib.load(checkpoint_path)
            self.threshold = int(checkpoint_path[-6:-4]) / 100

    def __preprocess_doc(self, text_layer: str) -> str:
        text_layer = text_layer.lower()
        document = re.sub(r'\W', ' ', text_layer)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        return document

    def __create_ngramm_list(self, text_layer: str, ngram_num: int) -> list:
        ngrams_list_text = ["".join(item) for item in ngrams(text_layer, ngram_num)]
        return ngrams_list_text

    def __get_probability(self, text_layer: str, ngram_num: int, model):
        text_layer = self.__preprocess_doc(text_layer)
        ngram_list = self.__create_ngramm_list(text_layer, ngram_num)
        probability_list = []
        for i in range(len(ngram_list) - 1):
            probability_list.append(model.score(ngram_list[i + 1], [ngram_list[i]]))
        probability = np.mean(probability_list)
        return probability

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int], parameters: Dict,
              feature_extractor: AbstractFeatureExtractor):
        documents = [self.__preprocess_doc(text) for i, text in enumerate(train_texts) if train_labels[i] == 1]
        documents = " ".join(documents)
        ngram_list = self.__create_ngramm_list(documents, self.ngram_num)
        train, vocab = padded_everygram_pipeline(3, [ngram_list])
        self.model.fit(train, vocab)

        eval_texts_tokenized = [self.__preprocess_doc(text) for text in eval_texts]
        eval_preds = [self.__get_probability(text_layer=text, model=self.model, ngram_num=self.ngram_num) for text in eval_texts_tokenized]

        f1_best = 0
        threshold_best = 0
        result = 0
        for threshold in np.arange(0, 1, 0.01):
            f1 = f1_score(eval_labels, np.where(np.array(eval_preds) > threshold, 1, 0), average='weighted')
            if f1 > f1_best:
                f1_best = f1
                threshold_best = threshold
                result = classification_report(eval_labels, np.where(np.array(eval_preds) > threshold, 1, 0), digits=3)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', str(self.ngram_num)+"-gram_"+['MLE', 'Laplace'][self.laplace]+"_thrshld_" + str('%.2f' % threshold_best)[-2:]) + ".pkl", "wb") as f:
            pickle.dump(self.model, f)
        self.threshold = threshold_best
        print("Best threshold:", threshold_best)
        print("Evaluation results")
        print(result)

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

    def predict(self, text_layer: List[str], parameters: Dict, feature_extractor: AbstractFeatureExtractor) -> List[int]:
        return np.where(np.array([self.__get_probability(text_layer=text, ngram_num=self.ngram_num, model=self.model) for text in text_layer]) > self.threshold, 1, 0).tolist()
