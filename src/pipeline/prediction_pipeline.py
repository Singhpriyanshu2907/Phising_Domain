import shutil
import os
import pandas as pd
import pickle
from src.autologger import logger

from src.exception import CustomException
from src.utils import load_object
import sys
from flask import request
from src.constant import *


from dataclasses import dataclass
        
        
@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "predicted_file.csv"
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self):
         self.prediction_pipeline_config = PredictionPipelineConfig()  # Initialize the configuration



    def save_input_files(self,input_file):

        try:
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = input_file
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)            
            input_csv_file.save(pred_file_path)
            return pred_file_path
        
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
            try:

                preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
                model_path=os.path.join('artifacts','model.pkl')              
                
                preprocessor = load_object(preprocessor_path)
                logger.info(f"Type of preprocessor: {type(preprocessor)}")
                model = load_object(model_path)
                

                transformed_x = preprocessor.transform(features)
                logger.info(f"Shape of transformed_x: {transformed_x.shape}")


                preds = model.predict(transformed_x)

                return preds

            except Exception as e:
                raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path):
        try:
            prediction_column_name = "phishing"
            input_dataframe = pd.read_csv(input_dataframe_path)

            # Assuming the input_dataframe contains the features for prediction
            # and you want to predict the 'phishing' column
            features = input_dataframe.drop(columns=prediction_column_name)

            # Call the predict method to get predictions
            predictions = self.predict(features)

            # Add predictions as a new column to the input_dataframe
            input_dataframe[prediction_column_name] = predictions

            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)

            logger.info("Predictions completed.")
        except Exception as e:
            raise CustomException(e, sys)

        

        
    def run_pipeline(self,input_file):
        try:
            input_csv_path = self.save_input_files(input_file)
            self.get_predicted_dataframe(input_csv_path)
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        