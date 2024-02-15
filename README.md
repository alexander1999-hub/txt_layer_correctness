# Text layer correctness experiments

In this repository you can run experiments with all methods described in paper.

## Requirements
<strong>This repository requires ```python==3.9```</strong><br>
You can create virtual environment with ```requirenets.txt```

## Dataset
Synthetic dataset for training and benchmark dataset will download automatically when running ```main.py```. <br>
All data will be stored in a ```./data``` folder that will also be created automatically. </br>

## Experiments
You can run experiments with XGBoost, Random Forest, Logistic Regression, N-Gram, Rubert with following command: <br>
```
python main.py
```
By default, it runs experiments with all methods using TF-IDF feature extractor <br>
 - You can select models for experiments by changing the corresponding list ```models``` in ```main.py``` <br>
 - You can also select feature extractor for experiments by changing the value of ```final_feature_extractor``` in ```main.py```

