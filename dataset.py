import csv
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


class Dataset:
    def __init__(self, synthetic_data_root: str, benchmark_data_root: str):
        self.synthetic_data_root = synthetic_data_root
        self.benchmark_data_root = benchmark_data_root

    def __get_files(self, path: Path) -> List[str]:
        path_all = []
        if path.is_dir():
            for subdir in path.iterdir():
                for subsubdir in subdir.iterdir():
                    path_all.append(str(subsubdir))
        else:
            print("Empty dir ", path)
        return path_all

    def __write_to_csv(self, filename: str, all_texts: List, correct_texts: List):
        with open(filename, mode="w+", encoding='utf-8') as w_file:
            file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r", escapechar='\\')
            file_writer.writerow(["text", "target", "file"])
            for i, file in tqdm(enumerate(all_texts)):
                with open(file, mode="r") as f:
                    text = f.read()

                label = 1 if file in correct_texts else 0
                file_writer.writerow([text.strip().replace('\n', ""), label, file.split('/')[-1]])

    def make_csv_files_synthetic(self):
        for part in ['train', 'val', 'test']:
            path_incorrect = os.path.join(self.synthetic_data_root, 'not_correct_'+part)
            path_correct = os.path.join(self.synthetic_data_root, 'correct_'+part)

            incorrect_texts = self.__get_files(Path(path_incorrect))
            correct_texts = self.__get_files(Path(path_correct))
            all_texts = correct_texts + incorrect_texts

            filename = "dataset_" + part + ".csv"
            self.__write_to_csv(filename=filename, all_texts=all_texts, correct_texts=correct_texts)

    def make_csv_files_benchmark(self):
        path_incorrect = os.path.join(self.benchmark_data_root, 'not_correct')
        path_correct = os.path.join(self.benchmark_data_root, 'correct')

        incorrect_texts = [os.path.join(path_incorrect, file) for file in os.listdir(path_incorrect)]
        correct_texts = [os.path.join(path_correct, file) for file in os.listdir(path_correct)]
        all_texts = correct_texts + incorrect_texts

        filename = "dataset_real.csv"
        self.__write_to_csv(filename=filename, all_texts=all_texts, correct_texts=correct_texts)

    def create_synthetic_dataset(self) -> Tuple:
        if not (os.path.isfile('dataset_train.csv') and os.path.isfile('dataset_val.csv') and os.path.isfile('dataset_test.csv')):
            self.make_csv_files_synthetic()

        dataset_train = pd.read_csv("dataset_train.csv")
        dataset_test = pd.read_csv("dataset_test.csv")
        dataset_val = pd.read_csv("dataset_val.csv")

        train_text = dataset_train['text'].astype('str')
        train_labels = dataset_train['target']
        val_text = dataset_val['text'].astype('str')
        val_labels = dataset_val['target']
        test_text = dataset_test['text'].astype('str')
        test_labels = dataset_test['target']

        return train_text, train_labels, val_text, val_labels, test_text, test_labels

    def create_benchmark_dataset(self) -> Tuple:
        if not os.path.isfile('dataset_real.csv'):
            self.make_csv_files_benchmark()

        dataset_real = pd.read_csv("dataset_real.csv")
        real_text = dataset_real['text'].astype('str')
        real_labels = dataset_real['target']

        return real_text, real_labels