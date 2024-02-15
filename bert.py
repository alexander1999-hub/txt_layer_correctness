import numpy as np
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from datasets import Dataset, load_metric
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from typing import Dict, List, Optional

from abstract_feature_extractor import AbstractFeatureExtractor
from abstract_model import AbstractModel


class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item
    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    return {'F1': f1}

class Bert(AbstractModel):
    def __init__(self, from_checkpoint: bool, checkpoint_path: Optional[str] = "fine-tune-bert"):
        self.model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')
        self.training_args = TrainingArguments(
            output_dir = './results_bert', #Выходной каталог
            num_train_epochs = 3, #Кол-во эпох для обучения
            per_device_train_batch_size = 8, #Размер пакета для каждого устройства во время обучения
            per_device_eval_batch_size = 8, #Размер пакета для каждого устройства во время валидации
            weight_decay = 0.01, #Понижение весов
            logging_dir = './logs', #Каталог для хранения журналов
            load_best_model_at_end = True, #Загружать ли лучшую модель после обучения
            learning_rate = 1e-5, #Скорость обучения
            evaluation_strategy ='epoch', #Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
            logging_strategy = 'epoch', #Логирование после каждой эпохи
            save_strategy = 'epoch', #Сохранение после каждой эпохи
            save_total_limit = 1,
            seed=21)
        self.__load_model(from_checkpoint, checkpoint_path)
        self.name = "RuBert"
        
    def __load_model(self, from_checkpoint: bool, checkpoint_path: str):
        if from_checkpoint and os.path.isfile(checkpoint_path):
            self.model = BertForSequenceClassification.from_pretrained(checkpoint_path)


    def __tokenize(self, text_sequence: List[str]):
        tokens = self.tokenizer.batch_encode_plus(
            text_sequence,
            max_length = 512,
            padding = 'max_length',
            truncation = True
        )
        return tokens


    def __create_dataset(self, texts: List[str], labels: List[int]):
        tokens = self.__tokenize(texts)
        return Data(tokens, labels)
    
    def __seed_all(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


    def __initialize_trainer(self, train_dataset, eval_dataset):
        self.trainer = Trainer(model=self.model,
                  tokenizer = self.tokenizer,
                  args = self.training_args,
                  train_dataset = train_dataset,
                  eval_dataset = eval_dataset,
                  compute_metrics = compute_metrics)
        
        return self.trainer

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: List[str], eval_labels: List[int], parameters: Dict,
              feature_extractor: AbstractFeatureExtractor):
        
        self.__seed_all(42)

        train_dataset = self.__create_dataset(train_texts, train_labels)
        eval_dataset = self.__create_dataset(eval_texts, eval_labels)

        self.__initialize_trainer(train_dataset, eval_dataset)
        
        self.trainer.train()

        model_path = "fine-tune-bert"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def test(self, test_texts: List[str], test_labels: List[int], parameters: Dict, feature_extractor: AbstractFeatureExtractor,
             measure_time: Optional[bool] = False):
        start = time.time()
        dataset = self.__create_dataset(test_texts, test_labels)
        pred = self.trainer.predict(dataset)
        end = time.time()
        labels = np.argmax(pred.predictions, axis = -1)
        print(classification_report(test_labels, labels, digits=3))
        if measure_time:
            print("Average predict time:", (end-start) / len(test_labels), "seconds")


    def predict(self, text_layer: str, parameters: Dict, feature_extractor: AbstractFeatureExtractor):
        clf = pipeline(task='text-classification', model=self.model, tokenizer=self.tokenizer)
        return clf(text_layer)
