import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.constant import *
from src.autologger import logger
from src.utils import export_collection_as_dataframe,Null_treatment,get_cols_with_zero_std_dev,load_object
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")

    raw_data_path: str = os.path.join("artifacts", "data.csv")

    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            df: pd.DataFrame = export_collection_as_dataframe(
                db_name="PWSkills", collection_name="PhisingInternship"
            )

            logger.info(f"Number of rows after ingestion: {len(df)}")

            logger.info("Removing Null values & Dropping col with 0 std dev")

            Null_treatment(df)

            df = get_cols_with_zero_std_dev(df)

            # Calculate the correlation matrix
            corr_matrix = df.corr().abs()

            # Create a boolean mask to identify columns with correlation greater than 0.90
            high_corr_cols = np.full(corr_matrix.shape, False, dtype=bool)
            high_corr_cols[np.triu_indices(len(high_corr_cols), k=1)] = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)] > 0.90

            # Get the column indices to drop
            cols_to_drop = np.where(high_corr_cols.any(axis=0))[0]

            # Drop the columns from the DataFrame
            df.drop(df.columns[cols_to_drop], axis=1, inplace=True)

            logger.info("Data Cleaned up")

            logger.info("Making dir & Splitting data into test & train")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logger.info(
                f"Ingested data from mongodb to {self.ingestion_config.raw_data_path}"
            )

            logger.info("Exited initiate_data_ingestion method of DataIngestion class")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

