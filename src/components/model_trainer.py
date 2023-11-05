import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from src.constant import *
from src.exception import CustomException
from src.autologger import logger
from src.utils import train_and_predict_models, read_yaml_file, save_object

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts")
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.75
    model_config_file_path = os.path.join('config', 'model.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'XG Boost': XGBClassifier(),
            'Extra Trees': ExtraTreeClassifier(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost Classifier': AdaBoostClassifier()
        }

    def get_best_model(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array):
        try:
            model_report = train_and_predict_models(x_train, y_train, x_test, y_test, self.models)
            print("Model training data shape:", x_train.shape)
            print("Model testing data shape:", x_test.shape)
            
            for model_name, model_score in model_report.items():
                print(f"{model_name} accuracy: {model_score:.2f}")
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score
        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(self, best_model_object, best_model_name, x_train, y_train) -> object:
        try:
            model_param_grid = read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(best_model_object, param_grid=model_param_grid, cv=3, n_jobs=-1, verbose=1)
            grid_search.fit(x_train, y_train)
            best_params = grid_search.best_params_
            print("Best parameters for", best_model_name, "are:", best_params)
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logger.info(f"Splitting training and testing input and target feature")
            x_train, y_train, x_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            
            logger.info(f"Training data shape: {x_train.shape}")
            logger.info(f"Testing data shape: {x_test.shape}")

            logger.info(f"Extracting model config file path")
            model_report = train_and_predict_models(x_train, y_train, x_test, y_test)

            for model_name, model_score in model_report.items():
                logger.info(f"{model_name} accuracy: {model_score:.2f}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = self.models[best_model_name]

            best_model = self.finetune_best_model(best_model, best_model_name, x_train, y_train)
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            logger.info(f"Best model name: {best_model_name}, accuracy: {best_model_score:.2f}")

            if best_model_score < 0.7:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")

            logger.info(f"Best found model on both training and testing dataset")

            logger.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)
            save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)
            
            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
