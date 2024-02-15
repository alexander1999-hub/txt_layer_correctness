from forest import RandomForest
from logreg import LogRegression
from xgb import XGBoost
from abstract_model import AbstractModel
from tf_idf import TFIDF
import os
import wget
import zipfile
from dataset import Dataset
from abstract_feature_extractor import AbstractFeatureExtractor
from custom_feature_extractor import CustomFeatureExtractor
from n_gram import NGram
from typing import Dict, List, Tuple
from bert import Bert


def train_and_test_one_model(model: AbstractModel, feature_extractor: AbstractFeatureExtractor,
                             dataset: Tuple[List[str], List[int], List[str], List[int], List[str], List[int]], parameters: Dict):
    train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels = dataset
    model.train(train_texts, train_labels, eval_texts, eval_labels, parameters, feature_extractor)
    print("Test results")
    model.test(test_texts, test_labels, parameters, feature_extractor, measure_time=False)


def benchmark_one_model(model: AbstractModel, feature_extractor: AbstractFeatureExtractor,
                        dataset: Tuple[List[str], List[int]], parameters: Dict):
    benchmark_texts, benchmark_labels = dataset
    print("Benchmark results")
    model.test(benchmark_texts, benchmark_labels, parameters, feature_extractor, measure_time=True)


def download_dataset(synthetic_data_url: str, benchmark_data_url: str, data_dir: str):
    if not os.path.isdir(data_dir):
        print("Start dataset downloading...") 
        os.makedirs(data_dir, exist_ok=True)

        synthetic_archive_path = os.path.join(data_dir, "archive_synthetic.zip")
        benchmark_archive_path = os.path.join(data_dir, "archive_benchmark.zip")

        wget.download(synthetic_data_url, synthetic_archive_path)
        wget.download(benchmark_data_url, benchmark_archive_path)

        with zipfile.ZipFile(synthetic_archive_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(synthetic_archive_path)

        with zipfile.ZipFile(benchmark_archive_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(benchmark_archive_path)

        assert os.path.isdir(data_dir)
        print(f"Training data downloaded to {data_dir}")  # noqa
    else:
        print(f"Use training data from {data_dir}")  # noqa
    
    return os.path.join(data_dir, 'data'), os.path.join(data_dir, 'benchmark_data')


if __name__ == '__main__':
    # download dataset
    (synthetic_data_dir, benchmark_data_dir) = download_dataset(synthetic_data_url="https://at.ispras.ru/owncloud/index.php/s/IKRqClmkPBqPRlI/download",
                     benchmark_data_url="https://at.ispras.ru/owncloud/index.php/s/2EvLfSQy3NF7AxJ/download",
                     data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

    # initialize all models
    forest = RandomForest(from_checkpoint=False)
    logreg = LogRegression(from_checkpoint=False)
    xgb = XGBoost(from_checkpoint=False)
    n_gram = NGram(from_checkpoint=False, ngram_num=1, laplace=True) # choose type of a model you want to use on n-grams
    rubert = Bert(from_checkpoint=False)

    # create list of models for experiments
    models = [xgb, forest, logreg, n_gram, rubert]

    # create dataset
    dataset_creator = Dataset(synthetic_data_root=synthetic_data_dir,
                              benchmark_data_root=benchmark_data_dir)
    synthetic_dataset = dataset_creator.create_synthetic_dataset()
    benchmark_dataset = dataset_creator.create_benchmark_dataset()

    # create feature extractors
    tf_idf_feature_extractor = TFIDF(train_data=synthetic_dataset[0]) # feature extractor based on TF-IDF
    custom_feature_extractor = CustomFeatureExtractor() # feature extractor based on heuristically chosen features

    # choose feature extractor for experiments
    final_feature_extractor = tf_idf_feature_extractor

    # start experiments
    for model in models:
        print("=========================================")
        print("Experiments with", model.name)
        print("Feature extractor:", final_feature_extractor.name)
        train_and_test_one_model(model=model,
                                 feature_extractor=final_feature_extractor,
                                 dataset=synthetic_dataset,
                                 parameters={})

        benchmark_one_model(model=model,
                            feature_extractor=final_feature_extractor,
                            dataset=benchmark_dataset,
                            parameters={})
        print("=========================================")
