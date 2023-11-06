from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.autologger import logger
import sys
import os
from src.pipeline.training_pipeline import TraininingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Phishing Domain Detection System"


@app.route("/train")
def train_route():
    try:
        if request.method == 'POST':
            input_file = request.files['file']  # Assuming 'file' is the name of the input file field in your HTML form
            train_pipeline = TraininingPipeline()
            train_pipeline.run_pipeline(input_file)  # Pass the input_file to the run_pipeline method
            return "Training Completed."

    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline()
            prediction_file_detail = prediction_pipeline.run_pipeline(request.files['file'])

            logger.info("Prediction completed. Downloading prediction file.")
            return send_file(
                prediction_file_detail.prediction_file_path,
                download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True
            )
        else:
            return render_template('upload_file.html')

    except Exception as e:
        raise CustomException(e, sys)
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug= True)