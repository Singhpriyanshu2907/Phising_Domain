import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.autologger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self, train_x, test_x, train_y):
        try:
            # Define the column names
            col = list(train_x.columns)

            # Feature selection using RandomForestClassifier on training data
            clf = RandomForestClassifier(random_state=42)
            clf.fit(train_x, train_y)

            feature_importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': col, 'Importance': feature_importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            cumulative_importance = feature_importance_df['Importance'].cumsum()
            selected_features_idx = (cumulative_importance <= 0.90).sum()
            selected_features = feature_importance_df.iloc[:selected_features_idx + 1]['Feature']

            # Feature selection using RFE on training data
            model = LogisticRegression(max_iter=5000)
            rfe = RFE(estimator=model, n_features_to_select=40)
            rfe = rfe.fit(train_x, train_y)
            imp_vars_RFE = list(train_x.columns[rfe.support_])

            # Convert selected_features and imp_vars_RFE to sets
            selected_features = set(selected_features)
            imp_vars_RFE = set(imp_vars_RFE)

            # Create a set that combines both sets
            combined_features_set = selected_features.intersection(imp_vars_RFE)

            # Select only the combined features in both training and test data
            x_train = train_x[list(combined_features_set)]
            x_test = test_x[list(combined_features_set)]

            # Define the column names
            cols = list(x_train.columns)

            # Create the preprocessing steps for numerical columns
            transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # Impute missing values if found any
                ('scaler', StandardScaler())  # Scale the data using StandardScaler
            ])

            # Create the column transformer
            preprocessor_X = ColumnTransformer(transformers=[
                ('num', transformer, cols)
            ])

            # Fit the ColumnTransformer on the training data
            transformed_train_data = pd.DataFrame(preprocessor_X.fit_transform(x_train),
            columns=preprocessor_X.get_feature_names_out())

            # Transform the test data
            transformed_test_data = pd.DataFrame(preprocessor_X.transform(x_test),
            columns=preprocessor_X.get_feature_names_out())

            return preprocessor_X,transformed_train_data,transformed_test_data

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "phishing"

            logger.info("Splitting X & Y variable")

            # Training dataframe
            train_x = train_df.drop(columns=[target_column_name], axis=1)
            train_y = train_df[target_column_name]

            # Testing dataframe
            test_x = test_df.drop(columns=[target_column_name], axis=1)
            test_y = test_df[target_column_name]

            logger.info("Applying Data transformation & Data Scaling")

            preprocessor, transformed_train_data, transformed_test_data = self.get_data_transformer_object(train_x, test_x, train_y)

            logger.info("Concatenating x & y variable under train_arr & test_arr")

            train_arr = np.c_[transformed_train_data, np.array(train_y)]
            test_arr = np.c_[transformed_test_data, np.array(test_y)]

            logger.info(f"Training data shape: {train_arr.shape}")
            logger.info(f"Testing data shape: {test_arr.shape}")

            logger.info("Saving pickle file for pre-processing")
            logger.info(f"Type of preprocessor: {type(preprocessor)}")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor) 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
