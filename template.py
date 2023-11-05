import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_transformation.py",
    f"src/components/model_trainer.py",
    f"src/utils.py",
    f"src/autologger.py",
    f"src/exception.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/prediction_pipeline.py",
    f"src/pipeline/training_pipeline.py",
    f"templates/upload_file.html",
    f"src/constant/__init__.py",
    f"Static/style.css",
    "config/config.yaml",
    "param.yaml",
    "app.py",
    "main.py",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    f"Research/"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")