"""
This module performs unit tests on the functions of the churn_libray.py

Author: Osman Kocas
Date: 08/02/2021
"""
import os
import logging
from churn_library import perform_eda, perform_feature_engineering
from churn_library import encoder_helper, train_models, import_data
from constants import CATEGORY_LST, RESPONSE, KEEP_COLUMNS

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    filename=r'./logs/test_churn_script_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import_data(import_data):
    """
    Tests import_data function
    Input:
            import_data: function under test
    Output:
            None
    """
    try:
        model_data = import_data(r"./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert model_data.shape[0] > 0
        assert model_data.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file is not of expected shape.")
        raise err


def test_perform_eda(perform_eda, model_data):
    """
    Tests perform eda function by checking if output files exists in the folder
    Input:
            perform_eda: function under test
    Output:
            None
    """
    model_data = perform_eda(model_data)
    for filename in ['churn_distribution', 'marital_status_distribution',
                     'customer_age_distribution', 'total_transaction_distribution',
                     'heatmap']:
        try:
            assert os.path.isfile(r"./images/eda/{}.png".format(filename))
            logging.info("Testing perform_eda: %s is created: SUCCESS", filename)
        except AssertionError as err:
            logging.error("Testing perform_eda function: CANNOT find %s", filename)
            raise err
        logging.info("Testing perform_eda function: SUCCESS")


def test_encoder_helper(encoder_helper, model_data, categories):
    """
    Tests perform_eda function by checking if output files exists in the folder
    Input:
            perform_eda: function under test
            model_data: pandas dataframe representing modeling dataset on which
                        encoder_helper will be tested
    Output:
            None
    """
    model_data = encoder_helper(model_data, categories, RESPONSE)

    for cat in categories:
        try:
            assert cat+'_'+RESPONSE in model_data
            logging.info("Testing encoder_helper: Encoding for %s is SUCCESSFUL", cat)
        except AssertionError as err:
            logging.error("Testing encoder_helper: Encoding for %s FAILED", cat)
            raise err


def test_perform_feature_engineering(perform_feature_engineering, model_data):
    """
    Tests perform_feature_engineering function by
        1. Checking if the binary response (label) is correctly created
        2. Checking if the test train splits have the expected shape
    Input:
            perform_feature_engineering: function under test
            model_data: pandas dataframe representing modeling dataset on which
                        feature engineering will be performed.
    Output:
            None
    """
    #number of observations in the dataset and expected number of final columns
    nobs = model_data.shape[0]
    ncols = len(KEEP_COLUMNS)
    nobs_train = int(nobs*0.7)
    nobs_test = nobs - nobs_train

    #apply feature engineering and obtain training and testing features and labels
    #features_and_labels = x_train, x_test, y_train, y_tests
    features_and_labels = perform_feature_engineering(model_data, RESPONSE)

    #expected test train data shapes for x_train, x_test, y_train, y_tests
    expected_shapes = [(nobs_train, ncols), (nobs_test, ncols), (nobs_train,), (nobs_test,)]

    #check if label is created
    try:
        assert RESPONSE in model_data
        logging.info("Testing perform_feature_engineering: %s is created SUCCESSFULLY", RESPONSE)
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED to create %s", RESPONSE)
        raise err

    #check if label is binary
    try:
        assert list(model_data[RESPONSE].unique()) == [0, 1]
        logging.info("Testing perform_feature_engineering: %s is binary", RESPONSE)
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED. %s is not binary", RESPONSE)
        raise err

    #check if the train test split data frames have the expected shape
    for i in range(0, 4):
        try:        
            #print (features_and_labels[i].shape)
            #print (expected_shapes[i])
            assert features_and_labels[i].shape == expected_shapes[i]
            logging.info("Testing perform_feature_engineering: Train, test split is SUCCESSUL")
        except AssertionError as err:
            logging.error("Testing perform_feature_engineering: FAILED to create train/test data")
            raise err


def test_train_models(train_models, x_train, y_train):
    """
    Tests train_models function by checking if the .pkl files exists under./models
    Input:
            perform_eda: function under test
            kwargs: keyword arguments passed to the function; x_train, y_train
    Output:
            None
    """
    train_models(x_train, y_train)
    try:
        assert os.path.isfile(r'./models/rfc_model.pkl')
        assert os.path.isfile(r'./models/logistic_model.pkl')
        logging.info("Testing train_models: Models are trained and saved SUCCESSFULLY")
    except AssertionError as err:
        logging.error("Testing train_models: CANNOT find models under the models directory")
        raise err


def run_all_tests():
    """
    Runs all the tests written above
    """
    test_import_data(import_data)

    model_data = import_data(r'./data/bank_data.csv')
    test_encoder_helper(encoder_helper, model_data, CATEGORY_LST)

    model_data = encoder_helper(model_data, CATEGORY_LST, RESPONSE)
    test_perform_eda(perform_eda, model_data)
    test_perform_feature_engineering(perform_feature_engineering, model_data)

    #create keyword arguments and test train_models function
    x_train, _, y_train, _ = perform_feature_engineering(model_data, RESPONSE)
    test_train_models(train_models, x_train, y_train)

if __name__ == "__main__":
    #run_all_tests
    run_all_tests()
