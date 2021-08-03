# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project provides codes for analyzing and predicting customer churn behavior on a dataset
from a bank. The codes are primarily from the churn_notebook.ipynb. They follow PEP8 styling guide and demonstrate coding, logging and testing best practices.
At a high level you the project delivers the following.

1. EDA for analyzing customer churn behavior on a bank's customer data.
2. A random forest and a logistic regression model both predicting customer churn probability for the bank. The models are serialized as a .pkl file and can be used on generating predictions on different, such as production, datasets.
3. A classification report for each of the two models.

## Installation
We recommend using Conda and run the following 
```
conda create --name myenv
source activate myenv
conda install --file requirements.txt
```

## Running the Scripts

In an ipython enviorement, run the following command to fit and save the models,
generate eda and classification reports, ruen

```
ipython churn_library.py
```
Or, on command window type

```
python3 churn_libary.py
```

The log file, churn_library_results.log, can be found under
the logs folder.
## Testing

To run the unit tests for the methods implemented in churn_library.py,
run the following command: 

```
ipython churn_script_logging_and_tests.py
```

The log file, test_churn_script_results.log, can be founder under logs folder.

