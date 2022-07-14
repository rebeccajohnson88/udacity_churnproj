#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import logging
import constants 
from constants import *
import churn_library
from churn_library import *


# In[ ]:


def test_import(import_data):
    '''
    test data import 
    '''
    try:
        df = import_data(PATH_DATA)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


# In[4]:


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        assert os.path.isdir(PATH_EDA_FIGS)
        logging.info("SUCCESS directory for images exists")
    except AssertionError as err:
        logging.error("ERROR inputted image directory does not exist")
        raise err
    try:
        assert len(features) > 0
        logging.info("SUCCESS some features for EDA")
    except AssertionError as err:
        logging.error("ERROR no features for EDA")
        raise err


# In[ ]:



def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    pass


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    logging.basicConfig(
    filename=PATH_LOGFILE,
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
    test_import(import_data)
    test_eda(perform_eda)

