This is a temporary anonymised repo to support the paper "Unsupervised Feature Based Algorithms for Time Series Extrinsic Regression" submitted to blind review for KDD 2023.

Datasets
--------

https://mega.nz/folder/0RpXVS7S#_vghTUbO7ZmJLssNQA7k4w <- monash problems (19)

https://mega.nz/folder/xRo2zKKL#FmADwhcEmCuhN6eEk-b-xA <- new problems (44)

Install
-------

pip install -r requirements.txt

Usage
-----

Run run_experiments.py with the following arguments:

1. path to the data directory

2. path to the results directory

3. the name of the model to run (see set_regressor.py, i.e. LR, DrCIF, CNN)

4. the name of the problem to run

5. the resample number to run (0 is base train/test split)

i.e. to run Covid3Month using linear regression on the base train/test split:

    run_experiments.py data/ results/new/ LR Covid3Month 0
