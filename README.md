# Diabetes Prediction Project

An end-to-end machine learning project for predicting diabetes risk using health indicators from the BRFSS2015 dataset from Kaggle ([data](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)). Created as part of the ML Zoomcamp from DataTalksClub.

## Overview
This repository contains code, configuration and utilities to train, evaluate and serve XGBoost models for diabetes prediction. It focuses on reproducibility and simple deployment with FastAPI.

## Features
- XGBoost-based binary classification (diabetes vs. no diabetes)
- Pydantic data validation for type safety
- FastAPI REST API for model inference
- Clear project layout for data, models, and notebooks
- Jupyter notebook for exploratory data analysis (EDA)

## Prerequisites
- Python 3.12
- uv package manager
- git
- docker (optional — see Docker section)

## Quick Start
1. Clone the repo
    ```
    git clone https://github.com/jacksonpaulp/diabetes_prediction_project.git
    cd diabetes_prediction_project
    ```
2. Create and activate virtual environment
    ```
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # or
    source .venv/bin/activate  # macOS/Linux
    ```
3. Install dependencies (uses uv / pyproject.toml)
    ```
    uv sync --locked
    ```

## Typical Workflow
- **Prepare data**: Place CSV files in `data/`
- **Explore data**: Open and run `notebooks/eda.ipynb`
- **Train model**:
  ```
  uv run .\src\train.py
  ```
- **Serve model**: Run FastAPI server
  ```
  uvicorn src.predict:app --reload
  ```

## Project Structure
```
diabetes_prediction_project/
├── data/                          # Raw and processed datasets
│   └── diabetes_012_health_indicators_BRFSS2015.csv
├── src/                           # Project source code
│   ├── train.py                   # Training script
│   └── predict.py                 # FastAPI prediction server (only this is copied into the container)
├── models/                        # Saved model artifacts
│   └── model.bin                  # Trained XGBoost model
├── notebooks/                     # Analysis and exploration
│   └── eda.ipynb                  # EDA notebook
├── pyproject.toml                 # Project/dependency metadata
├── uv.lock                        # Locked dependencies for uv
├── .python-version                # Python version pin
├── Dockerfile                     # Container image definition
├── README.md                      # This file
└── .dockerignore                  # Files to exclude from image (recommended)
```

## Training
Run the training script:
```
uv run .\src\train.py
```
This will:
- Load data from `data/diabetes_012_health_indicators_BRFSS2015.csv`
- Train an XGBoost classifier with optimized hyperparameters
- Save the model to `models/model.bin`

## API Inference
Start the FastAPI server:
```
uvicorn src.predict:app --reload
```
Then send POST requests to `http://localhost:8000/predict` with patient health indicators.

Example request body:
```json
{
  "highbp": 1,
  "highchol": 1,
  "cholcheck": 1,
  "smoker": 0,
  "stroke": 0,
  "heartdiseaseorattack": 0,
  "physactivity": 1,
  "fruits": 1,
  "veggies": 1,
  "hvyalcoholconsump": 0,
  "anyhealthcare": 1,
  "nodocbccost": 0,
  "diffwalk": 0,
  "sex": 1,
  "education": 3,
  "income": 4,
  "age": 5,
  "bmi": 25.5,
  "genhlth": 3,
  "menthlth": 0,
  "physhlth": 0
}
```

## Data
The project uses the BRFSS2015 Diabetes Health Indicators dataset. The target variable `diabetes_012` is binary-encoded:
- 0 = No diabetes or Pre-diabetes
- 1 = Diabetes

## Model
- **Algorithm**: XGBoost Classifier
- **Evaluation Metric**: Recall (focus on minimizing false negatives)

## Docker

Notes
- The project Dockerfile is set up to copy only `src/predict.py` and the trained model (`models/model.bin`) into the image to keep it small and focused on serving.

Build image
```
docker build -t diabetes_prediction:latest .
```

```
docker run -p 8000:8000 diabetes_prediction:latest
```

## Contact
For questions or issues, refer to the repository issue tracker or contact the project owner.
