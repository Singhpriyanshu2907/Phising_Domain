if __name__=="__main__":
    obj=DataIngestion()
    train_set,test_set=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_path = data_transformation.initiate_data_transformation(train_set,test_set)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path))