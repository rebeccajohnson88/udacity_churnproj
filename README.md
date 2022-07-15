# Predict Customer Churn 

This is the first project of the [Udacity ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) which tests lessons in the Clean Code Principles module.

## Project Description

In this project, our goal is to (1) predict which credit card customers are most likely to "churn" (attrit from using the credit card services) and (2) explore top predictors of churn using a set of financial and demographic variables.

Beyond the substantive focus, the project tests clean code principles, including logging and writing tests for user-defined functions.

## Files and data description

Before any of the files are run, here are the files in the root directory and directory structure that needs to be set up locally:


```bash
├── Guide.ipynb
├── README.md
├── churn_library.py
├── churn_script_logging_and_tests.py
├── constants.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   └── results
├── logs
├── models
```

The data are available here: [data/bank_data.csv](https://github.com/rebeccajohnson88/udacity_churnproj/blob/main/data/bank_data.csv)

The data is originally [from this Kaggle page](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers). The data has rows for ~10k credit card customers and contains demographic and financial information.

## Running Files

The order to run the files is the following, with the resulting expected outputs:

1. Open up terminal and run `pip install requirements.txt`. If this works, you should either see installation messages or messages that the requirements are already satisfied.
2. Run the main model estimation script: `python churn_library.py` that generates various EDA and predictive model results. This calls the following [constants file](https://github.com/rebeccajohnson88/udacity_churnproj/blob/main/constants.py), which should be modified if pathnames, the label name, or other constant names change.
  - pylint score: 8.09/10
3. Run the logging and testing script: `python churn_script_logging_and_tests.py`. You can check the results of the tests in './logs/churn_library.log' and you can compare your log results to the log in the repo.
  - pylint score: 8.15/10

The scripts in `archive` are not to be run and instead are scripts either provided as the base of the project (e.g., `churn_notebook.ipynb`) or older requirements.tx that came with the project 

The directory structure that results from a fully executed project is as follows:






