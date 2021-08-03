"""
This module contains methods used in predicting 
and analyzing customer churn on a bank's data.

Author: Osman Kocas
Date: 08/02/2021
"""

import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from constants import FIG_SIZE, CATEGORY_LST, PARAM_GRID, RESPONSE, KEEP_COLUMNS, RANDOM_STATE

sns.set()

logging.basicConfig(
    filename=r'./logs/churn_library_results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(path):
    """
    returns dataframe for the csv found at pth

    Input:
            path: a path to the csv
    Output:
            data: pandas dataframe
    """
    try:
        model_data = pd.read_csv(path)
        logging.info('Data loaded from %s : SUCCESS', path)
    except FileNotFoundError:
        logging.error('Data load from %s failed. Please check file at path', path)
        return "Cannot find file. Please check the file path."
    return model_data


def perform_eda(model_data):
    """
    Perform eda on data and save figures to images folder
    Input:
            model_data: pandas dataframe, modeling dataset

    output:
            None
    """
    #churn  and customer age distribution
    for col in ['Churn', 'Customer_Age']:
        plt.figure(figsize=FIG_SIZE)
        model_data[col].hist()
        plt.savefig(r'./images/eda/{}_distribution.png'.format(col.lower()))
        plt.close()

    #marital status distribution
    plt.figure(figsize=FIG_SIZE)
    model_data['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(r'./images/eda/marital_status_distribution.png', dpi=300, bbox_inches = "tight")
    plt.close()

    #total transaction distribution
    plt.figure(figsize=FIG_SIZE)
    sns.distplot(model_data['Total_Trans_Ct'])
    plt.savefig(r'./images/eda/total_transaction_distribution.png', dpi=300, bbox_inches = "tight")
    plt.close()

    #correlation matrix heatmap
    plt.figure(figsize=FIG_SIZE)
    sns.heatmap(model_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(r'./images/eda/heatmap.png', dpi=300, bbox_inches = "tight")
    plt.close()


def encoder_helper(model_data, category_lst, response):
    """
    Helper function for
        1. Turn each categorical column into a new column with
           propotion of churn for each category - associated with cell
           15 from the notebook
        2. Creating target variable, 'Churn'

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            model_data: pandas dataframe with new encoded columnn containing propoportion
                        of churn of each category
    """
    #define response, i.e. churn
    try:
        model_data[RESPONSE] = model_data['Attrition_Flag'].apply(lambda val: 0
                                                                  if val == "Existing Customer"
                                                                  else 1)
        logging.info('Created %s label SUCCESSFULLY', RESPONSE)
    except Exception as err:
        logging.info('ERROR: Cannot create %s label', RESPONSE)
        raise err

    for cat in category_lst:
        try:
            model_data[cat+'_'+response] = model_data.groupby(cat)[response].transform(lambda x:
                                                                                       x.mean())
            logging.info('Created encoded feature for %s SUCCESSFULLY', cat)
        except Exception as err:
            logging.error('ERROR: Cannot create encoded feature for %s', cat)
            raise err

    return model_data


def perform_feature_engineering(model_data, response):
    """
    Performs feature engineering by
        1. Features encoding the propoportion of churn of each category
           in CATEGORY_LST
        2. Performing train/test split
    Input:
              model_data: pandas dataframe with modeling data
              response: string of response name [optional argument that
                        could be used for naming variables or index y column]

    Output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    #create encoded features
    model_data = encoder_helper(model_data, category_lst=CATEGORY_LST, response=RESPONSE)

    #keep columns needed
    x_all = model_data[KEEP_COLUMNS]
    #split into training ante test sets
    x_train, x_test, y_train, y_test = train_test_split(x_all, model_data[response],
                                                        test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train, y_test, predictions):
    """
    Produces classification report for training and testing results and stores report as image
    in images folder
    Input:
            y_train: training response values
            y_test:  test response values
            predictions: train and test predicitons for logistic and random forest models

    Output:
             None
    """
    #unpack predictions for logistic regression and random forest models
    y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf = predictions
    try:
        _save_classification_report('Random Forest', y_train, y_test,
                                    y_train_preds_rf, y_test_preds_rf)
        logging.info('Generated classification report for the Random Forest model')
    except Exception as err:
        logging.error('Cannot generate classification report for the Random Forest model')
        raise err

    try:
        _save_classification_report('Logistic Regression', y_train, y_test,
                                    y_train_preds_lr, y_test_preds_lr)
        logging.info('Generated classification report for the Logistic Regression model')
    except Exception as err:
        logging.error('Cannot generate classification report for the Logistic Regression model')
        raise err


def feature_importance_plot(model, x_data, output_path):
    """
    Creates and stores the feature importances in output_path
    Input:
            model: model object containing feature_importances_
            x_data: pandas dataframe with model features
            output_path: path to store the figure

    Output:
             None
    """
    # Calculate feature importances
    try:
        importances = model.feature_importances_
        logging.info('Feature importances are obtained SUCCESSFULLY')
    except AttributeError as err:
        logging.error('ERROR: Choose a model with feature importance attribute')
        raise err
    except Exception as exc:
        logging.error('ERROR: Cannot get feature importances. Make sure the model is fit')
        raise exc
    else:

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [x_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(x_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        #save plot
        plt.savefig(output_path+'/rfc_feature_importance.png', dpi=300, bbox_inches = "tight")
        plt.close()
        logging.info('Feature importance plot was created')


def train_models(x_train, y_train, return_models=False, retrain=True):
    """
    Trains both models and returns trained models in a dictionary
    Input:
            x_train: pandas dataframe; feature data frame for training data
            y_train: pandas series; target values for training data
            return_models: Boolean. Set True to keep models in memory
    Output:
            trained_models: dict {"rfc_model":cv_rfc,"lr_model":lrc_model}
    """
    #initialize the classifier
    logging.info('Fitting classifiers..')
    #classifiers to be fitted
    rfc_model = RandomForestClassifier(random_state=RANDOM_STATE)
    lrc_model = LogisticRegression()

    #perform grid search for random forest,rfc_model and fit the best model
    cv_rfc = GridSearchCV(estimator=rfc_model, param_grid=PARAM_GRID, cv=5)
    logging.info('Performing Grid Search for the Random Forest Model...')
    cv_rfc.fit(x_train, y_train)

    #fit the logistic regressio,lr model
    lrc_model.fit(x_train, y_train)
    logging.info('Both Models are fit SUCCESSFULLY')
    try:
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc_model, './models/logistic_model.pkl')
        logging.info('Models are saved SUCCESSFULLY')
    except Exception as err:
        logging.error("ERROR: Cannot save models. A problem occcured.")
        raise err
    return (lrc_model, cv_rfc.best_estimator_) if return_models else None


def generate_roc_curve(trained_models, x_test, y_test):
    """
    Generates and saves ROC Curves for the random forest and logistic regression classifiers
    on the test datasets
    Input:
            trained_models: a tuple of trained models; trained_models[0] is lrc_model,
                            trained_models[1] is the rfc_model
            x_test: pandas series; target values for training data
    Output:
            None
    """
    #save plots
    try:
        plt.cla()
        plt.figure(figsize=(15, 8))
        lrc_plot = plot_roc_curve(trained_models[0], x_test, y_test)
        axis = plt.gca()
        _ = plot_roc_curve(trained_models[1], x_test, y_test, ax=axis, alpha=0.8)
        lrc_plot.plot(ax=axis, alpha=0.8)
        plt.savefig(r'./images/roc_curve_result.png', dpi=300, bbox_inches = "tight")
        plt.close()
        logging.info('ROC Curve Plot is saved')
    except Exception as err:
        logging.error('Cannot find one or more models. Please ensure models are trained.')
        raise err


def _save_classification_report(model, y_train, y_test, y_train_preds, y_test_preds):
    """
    Private helper method to generate and save classification report for a given model
    Input:
            model: string, the model name
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from model
            y_test_preds: testing predictions from model

    Output:
            None

    """
    plt.cla()
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(model+' Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10}, fontproperties='monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(model+ ' Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10}, fontproperties='monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'./images/results/'+model+'_results.png', dpi=300, bbox_inches = "tight")

def run_all():
    """
    Method to run all the functions in the main block
    Input:
            None
    Output:
            None
    """
    #read csv file into data frame
    model_data = import_data(r'./data/bank_data.csv')

    #feature_engineering
    x_train, x_test, y_train, y_test = perform_feature_engineering(model_data, RESPONSE)

    #perform eda
    perform_eda(model_data)

    #train models
    models = train_models(x_train, y_train, return_models=True, retrain=False)
    logistic_model = models[0]
    rfc_model = models[1]

    #random forest predictions
    y_train_preds_rfc = rfc_model.predict(x_train)
    y_test_preds_rfc = rfc_model.predict(x_test)

    #logistic regression predictions
    y_train_preds_lrc = logistic_model.predict(x_train)
    y_test_preds_lrc = logistic_model.predict(x_test)

    #keep predictions in a list
    predictions = [y_train_preds_lrc, y_test_preds_lrc, y_train_preds_rfc, y_test_preds_rfc]

    #generate roc_plot
    generate_roc_curve(models, x_test, y_test)

    #generate feature importance plot
    feature_importance_plot(rfc_model, x_train, r'./images/results/')

    #generate classification report
    classification_report_image(y_train,
                                y_test,
                                predictions)
if __name__ == '__main__':
    #run all functions
    run_all()
    logging.shutdown()
