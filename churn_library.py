
"""
Functions for Udacity project 1: customer churn

Author: Rebecca A. Johnson
Date: 071422

"""


# import libraries
import os

import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib


import constants
from constants import (PATH_DATA, PATH_EDA_FIGS, 
PATH_RESULTS_FIGS, PATH_MODELS, LABEL_NAME, PARAM_GRID_RF,
COLS_TOKEEP)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    ## read in data
    df_use = pd.read_csv(pth)

    # construct binary churn indicator
    df_use['Churn'] = df_use['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df_use


def visualize_cat(df_use, one_colname):
    '''
    visualize categorical variables
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = plt.figure(figsize=(20, 10))
    df_use[one_colname].value_counts('normalize').plot(kind='bar')
    one_fig.savefig(PATH_EDA_FIGS + one_colname + ".png")
    plt.close()


def visualize_numeric(df_use, one_colname):
    '''
    visualize numeric variables
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = plt.figure(figsize=(20, 10))
    df_use[one_colname].hist()
    one_fig.savefig(PATH_EDA_FIGS + one_colname + ".png")
    plt.close()


def visualize_numeric_bivar(df_use, one_colname):
    '''
    create histogram summarizing relationship between churn
    and distribution of each numeric var
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = sns.displot(df_use, x=one_colname, hue=LABEL_NAME,
                          multiple="dodge")
    one_fig.savefig(PATH_EDA_FIGS + one_colname + "_bivar_wchurn.png")


def perform_eda(df_use):
    '''
    perform eda--univar and bivariate plots---on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # first, separate features into cat versus numeric
    non_features = ["Unnamed: 0", "Attrition_Flag", "CLIENTNUM"]
    features = [col for col in df_use.columns if col not in non_features]
    cat_columns = df_use[features].dtypes[df_use[features].dtypes == "object"].index
    quant_columns = df_use[features].dtypes[(df_use[features].dtypes == "int64") | (
        df_use[features].dtypes == "float64")].index

    # then, iterate through each type and produce and save figures
    cat_plots = [visualize_cat(df_use, one_col) for one_col in cat_columns]
    numeric_plots = [
        visualize_numeric(
            df_use,
            one_col) for one_col in quant_columns]
    bivar_plots = [
        visualize_numeric_bivar(
            df_use,
            one_col) for one_col in quant_columns if one_col != LABEL_NAME]

    # return categorical columns for use later for encoding
    return cat_columns


def encoder_helper(df_use, category_lst, response="_Churn"):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    # iterate through categorical cols and add to data with underscore _churn
    for one_col in category_lst:
        group_dict = df_use.groupby(one_col).mean()[LABEL_NAME]
        df_use[one_col + response] = df_use[one_col].map(group_dict)

    # return entire dataframe
    return df_use


def perform_feature_engineering(df_use, keep_cols, response=LABEL_NAME):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # separate into X and Y
    y = df_use[response]

    # subset to those cols
    X = df_use[[col for col in df_use.columns if col in keep_cols]].copy()

    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # return
    return(X_train, X_test, y_train, y_test)


def classification_report_image(modeldict):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    for key, value in modeldict.items():
        plt.rc('figure', figsize=(10, 5))
        plt.text(0.01, 0.45, str(key + ' Train: '), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(value[0], value[1])),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.9, str(key + ' Test: '), {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.5, str(classification_report(value[2], value[3])),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(PATH_RESULTS_FIGS + key + "_classificationreport.png")
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model dictionary where a value is the model to find fi/coef
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # first, check whether rf or lrc
    for key, value in model.items():

        # 1. Set up conditional logic to deal with RF fi
        # versus LR coef
        if key == "Random Forest":
            imp = value[4].feature_importances_
            xlabel = "Feature importances for: " + key
        else:
            imp = value[4].coef_[0]
            xlabel = "Coef for: " + key

        # 2. create dataframe w/ info for plot
        df_forplot = pd.DataFrame({'imp_coef': imp,
                                   'name': X_data.columns}).sort_values(by='imp_coef',
                                                                        ascending=False)

        # 3. Plot
        fi_plot = sns.barplot(x="imp_coef", y="name", data=df_forplot)
        fi_plot.set(ylabel="Feature", xlabel=xlabel)
        fi_plot.figure.savefig(output_pth + key + "_fiorcoef.png")

        # 4. Remove imp
        del imp


def train_models(X_train, X_test, y_train, y_test, estimate_new):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # 1. initialize models to train
    # model 1: random forest w/ CV grid search
    rfc = RandomForestClassifier(random_state=42)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=PARAM_GRID_RF, cv=5)

    # model 2: LR
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # 2. fit models
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # 3. for random forest, get top-performer
    # from grid search
    rfc_best = cv_rfc.best_estimator_

    # 4. save models
    if estimate_new:
        joblib.dump(rfc_best, PATH_MODELS + "rfc_model.pkl")
        joblib.dump(lrc, PATH_MODELS + "logistic_model.pkl")
    else:
        rfc_best = joblib.load(PATH_MODELS + "rfc_model.pkl")
        lrc = joblib.load(PATH_MODELS + "logistic_model.pkl")

    # 5. generate predictions
    y_train_preds_rf, y_test_preds_rf = [rfc_best.predict(x) for x
                                         in [X_train, X_test]]
    y_train_preds_lr, y_test_preds_lr = [lrc.predict(x) for x
                                         in [X_train, X_test]]

    # 6. Create dictionary to store models to avoid repetitive code
    # across mods
    models_tosummarize = {
        'Random Forest': [
            y_train,
            y_train_preds_rf,
            y_test,
            y_test_preds_rf,
            rfc_best],
        'Logistic Regression': [
            y_train,
            y_train_preds_lr,
            y_test,
            y_test_preds_lr,
            lrc]}

    # 7. Plot and save ROC curves
    # note: adapted code from here:
    # https://www.statology.org/plot-roc-curve-python/
    for key, value in models_tosummarize.items():
        y_pred_proba = value[4].predict_proba(X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        # create ROC curve
        plt.plot(fpr, tpr, label=key + " model AUC=" + str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig(PATH_RESULTS_FIGS + key + "_roc_curve.png")
        plt.close()

    # 8. Calculate classification report and save images
    classification_report_image(modeldict=models_tosummarize)

    # 9. Plot feature importances or coef and save images
    feature_importance_plot(model=models_tosummarize,
                            X_data=X_train,
                            output_pth=PATH_RESULTS_FIGS)


if __name__ == '__main__':
    df_output1 = import_data(PATH_DATA)
    print('Imported data')
    cat_columns = perform_eda(df_use=df_output1)
    print("Created plots")
    df_output2 = encoder_helper(df_use=df_output1, category_lst=cat_columns)
    print("Encoded cat vars")
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_use=df_output2, keep_cols = COLS_TOKEEP)
    print("Prepped data for model estimation")
    train_models(X_train, X_test, y_train, y_test, estimate_new=False)
    print("Estimated models and stored results")
