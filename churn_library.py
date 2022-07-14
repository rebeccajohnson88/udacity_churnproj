
#################################################
# Functions for Udacity project 1: customer churn
# Author: Rebecca A. Johnson
# Date: 071422
#################################################


# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import plot_roc_curve, classification_report

import constants  
from constants import *


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    ## read in data
    df = pd.read_csv(pth)

    ## construct binary churn indicator
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if 
                                        val == "Existing Customer" else 1)

    return(df)

def visualize_cat(df, one_colname):
    '''
    visualize categorical variables 
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = plt.figure(figsize=(20,10)) 
    df[one_colname].value_counts('normalize').plot(kind='bar')
    one_fig.savefig(PATH_EDA_FIGS + one_colname + ".png")
    plt.close()
    return None

def visualize_numeric(df, one_colname):
    '''
    visualize numeric variables 
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = plt.figure(figsize=(20,10)) 
    df[one_colname].hist()
    one_fig.savefig(PATH_EDA_FIGS + one_colname + ".png")
    plt.close()
    return None

def visualize_numeric_bivar(df, one_colname):
    '''
    create histogram summarizing relationship between churn
    and distribution of each numeric var
    input:
            df: pandas dataframe
            one_colname: string with name of column to visualize

    output:
            figure to pass back to main eda function
    '''
    one_fig = sns.displot(df, x=one_colname, hue= LABEL_NAME, 
                          multiple="dodge")
    one_fig.savefig(PATH_EDA_FIGS + one_colname + "_bivar_wchurn.png")
    return None


def perform_eda(df):
    '''
    perform eda--univar and bivariate plots---on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    
    ## first, separate features into cat versus numeric
    non_features = ["Unnamed: 0", "Attrition_Flag", "CLIENTNUM"]
    features = [col for col in df.columns if col not in non_features]
    cat_columns = df[features].dtypes[df[features].dtypes == "object"].index
    quant_columns = df[features].dtypes[(df[features].dtypes == "int64") |
                                    (df[features].dtypes == "float64")].index
    
    ## then, iterate through each type and produce and save figures
    [visualize_cat(df, one_col) for one_col in cat_columns]
    [visualize_numeric(df, one_col) for one_col in quant_columns]
    [visualize_numeric_bivar(df, one_col) for one_col in quant_columns
    if one_col != LABEL_NAME]
    
    ## return categorical columns for use later for encoding
    return cat_columns


def encoder_helper(df, category_lst, response = "_Churn"):
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
    
    ## iterate through categorical cols and add to data with underscore _churn
    for one_col in category_lst:
        group_dict = df.groupby(one_col).mean()[LABEL_NAME]
        df[one_col + response] = df[one_col].map(group_dict)
    
    ## return entire dataframe
    return(df)


def perform_feature_engineering(df, response = LABEL_NAME):
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
    
    ## separate into X and Y
    y = df[response]
    
    ## list of cols to keep
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    ## subset to those cols
    X = df[[col for col in df.columns if col in keep_cols]].copy()
    
    ## create train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, 
                                                        random_state=42)
    
    ## return
    return(X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                modeldict):
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
                         fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(value[0], value[1])), 
         {'fontsize': 10}, fontproperties = 'monospace') 
        plt.text(0.01, 0.9, str(key + ' Test: '), {'fontsize': 10}, 
                 fontproperties = 'monospace')
        plt.text(0.01, 0.5, str(classification_report(value[2], value[3])), 
                 {'fontsize': 10}, fontproperties = 'monospace') 
        plt.axis('off')
        plt.savefig(PATH_RESULTS_FIGS + key + "_classificationreport.png")
        plt.close()
        
    return None


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
    
    ## first, check whether rf or lrc
    for key, value in model.items():
        
        ## 1. Set up conditional logic to deal with RF fi
        ## versus LR coef
        if key == "Random Forest":
            imp = value[4].feature_importances_
            xlabel = "Feature importances for: " + key
        else:
            imp = value[4].coef_[0]
            xlabel = "Coef for: " + key
            
        ## 2. create dataframe w/ info for plot
        df_forplot = pd.DataFrame({'imp_coef': imp,
                    'name': X_data.columns}).sort_values(by = 'imp_coef',
                     ascending = False)
        
        ## 3. Plot
        fi = sns.barplot(x="imp_coef", y="name", data= df_forplot)
        fi.set(ylabel="Feature", xlabel= xlabel)
        fi.figure.savefig(output_pth + key + "_fiorcoef.png")

        ## 4. Remove imp
        del imp
        
    ## 5. return nothing
    return None 


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
    
    ## 1. initialize models to train
    ### model 1: random forest w/ CV grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    
    ### model 2: LR 
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    
    ## 2. fit models 
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    
    ## 3. for random forest, get top-performer
    ## from grid search
    rfc_best = cv_rfc.best_estimator_
    
    ## 4. save models
    if estimate_new:
        joblib.dump(rfc_best, PATH_MODELS + "rfc_model.pkl")
        joblib.dump(lrc, PATH_MODELS + "logistic_model.pkl")
    else:
        rfc_best = joblib.load(PATH_MODELS + "rfc_model.pkl")
        lrc = joblib.load(PATH_MODELS + "logistic_model.pkl")
        
    ## 5. generate predictions
    y_train_preds_rf, y_test_preds_rf = [rfc_best.predict(x) for x
                                    in [X_train, X_test]]
    y_train_preds_lr, y_test_preds_lr = [lrc.predict(x) for x
                                    in [X_train, X_test]]

    ## 6. Create dictionary to store models to avoid repetitive code 
    ## across mods
    models_tosummarize = {'Random Forest': [y_train, y_train_preds_rf,
                                            y_test, y_test_preds_rf, rfc_best],
                         'Logistic Regression': [y_train, y_train_preds_lr,
                                                y_test, y_test_preds_lr, lrc]}

    ## 7. Plot and save ROC curves
    ## note: adapted code from here: https://www.statology.org/plot-roc-curve-python/
    for key, value in models_tosummarize.items():
        y_pred_proba = value[4].predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)

        #create ROC curve
        plt.plot(fpr,tpr,label= key + " model AUC="+str(auc))
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.savefig(PATH_RESULTS_FIGS + key + "_roc_curve.png")
        plt.close()

    
    ## 8. Calculate classification report and save images 
    classification_report_image(y_train,
                           y_test,
                        y_train_preds_lr,
                        y_train_preds_rf,
                        y_test_preds_lr,
                        y_test_preds_rf, models_tosummarize)

    ## 9. Plot feature importances or coef and save images
    feature_importance_plot(model = models_tosummarize, 
                            X_data = X_train, 
                            output_pth = PATH_RESULTS_FIGS)


    ## return none
    return None 

if __name__ == '__main__':
    df = import_data(PATH_DATA)
    print('Imported data')
    cat_columns = perform_eda(df = df)
    print("Created plots")
    df = encoder_helper(df = df, category_lst = cat_columns)
    print("Encoded cat vars")
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    print("Prepped data for model estimation")
    train_models(X_train, X_test,  y_train, y_test, estimate_new = False)
    print("Estimated models and stored results")

