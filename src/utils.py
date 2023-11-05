import os
import sys
import yaml
import boto3
import dill
import pickle
import numpy as np 
import pandas as pd
from pymongo import MongoClient
from pathlib import Path
import re
from sklearn.tree import ExtraTreeClassifier,DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.autologger import logger


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    


def treat_outliers(df, lower_quantile=0.01, upper_quantile=0.99):


    treated_df = pd.DataFrame()

    for column in df.columns:
        lower_threshold = df[column].quantile(lower_quantile)
        upper_threshold = df[column].quantile(upper_quantile)

        treated_column = df[column].clip(lower_threshold, upper_threshold)
        treated_df[column] = treated_column

    return treated_df


def export_collection_as_dataframe(collection_name, db_name):
    try:
        mongo_client = MongoClient("mongodb+srv://priyanshus549:Priyanshu2907@phisingdomain.uumkuaq.mongodb.net/")

        collection = mongo_client[db_name][collection_name]

        df = pd.DataFrame(list(collection.find({})))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        return df
    except Exception as e:
        raise CustomException(e, sys)
    


def upload_file(from_filename, to_filename, bucket_name):
    try:
        s3_resource = boto3.resource("s3")

        s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

    except Exception as e:
        raise CustomException(e, sys)

## Function to treat missing values

def Null_treatment(x):
    for col in x.columns:
        if x[col].dtype == 'object':
         x[col].fillna(x[col].mode()[0],inplace=True)
        else:
           x[col].fillna(x[col].median(),inplace=True)
    return(x)



## Below function will be used to drop Variables with std_dev 0 as those variable wont contribute much in predictions

def get_cols_with_zero_std_dev(df: pd.DataFrame):
    """
    Returns a list of columns names which are having zero standard deviation.
    """
    cols_to_drop = []
    num_cols = [col for col in df.columns if df[col].dtype != 'O']  # numerical cols only
    for col in num_cols:
        if df[col].std() == 0:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)




def train_and_predict_models(train_x, train_y, test_x, test_y):
    # Create the model list
    model_list = {
        'Random Forest': RandomForestClassifier(),
        'XG Boost': XGBClassifier(),
        'Extra Trees': ExtraTreeClassifier(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost Classifier': AdaBoostClassifier()
    }

    results = {}

    for model_name, model in model_list.items():
        model.fit(train_x, train_y)

        # Make predictions on training and testing data
        train_pred = model.predict(train_x)
        test_pred = model.predict(test_x)

        # Calculate accuracy for training and testing data
        train_accuracy = r2_score(train_y, train_pred)
        test_accuracy = r2_score(test_y, test_pred)

        results[model_name] = test_accuracy

    return results

def read_yaml_file(filename):
    try:
        with open(filename, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys) from e

def read_schema_config_file(self) -> dict:
    try:
        schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

        return schema_config

    except Exception as e:
        raise CustomException(e, sys) from e