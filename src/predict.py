import os

import pickle
from typing import Literal, Optional
from pydantic import BaseModel, Field

import pandas as pd


from fastapi import FastAPI
import uvicorn

MODELNAME = os.path.join('models', 'model.bin')

class Patient(BaseModel):
    """Pydantic model for diabetes health indicators"""
    
    highbp: int = Field(..., description="High blood pressure indicator")
    highchol: int = Field(..., description="High cholesterol indicator")
    cholcheck: int = Field(..., description="Cholesterol check indicator")
    bmi: float = Field(..., description="Body Mass Index")
    smoker: int = Field(..., description="Smoker status")
    stroke: int = Field(..., description="Stroke history")
    heartdiseaseorattack: int = Field(..., description="Heart disease or attack history")
    physactivity: int = Field(..., description="Physical activity indicator")
    fruits: int = Field(..., description="Fruit consumption indicator")
    veggies: int = Field(..., description="Vegetable consumption indicator")
    hvyalcoholconsump: int = Field(..., description="Heavy alcohol consumption indicator")
    anyhealthcare: int = Field(..., description="Any healthcare coverage")
    nodocbccost: int = Field(..., description="Could not see doctor due to cost")
    genhlth: int = Field(..., description="General health status")
    menthlth: int = Field(..., description="Mental health days")
    physhlth: int = Field(..., description="Physical health days")
    diffwalk: int = Field(..., description="Difficulty walking indicator")
    sex: int = Field(..., description="Sex: 0=Female, 1=Male")    
    age: int = Field(..., description="Age group")
    education: int = Field(..., description="Education level")
    income: int = Field(..., description="Income level")

    


class PredictDiabetes(BaseModel):
    diabetes_probability: float
    diabetes: bool


app = FastAPI(title="diabetes-prediction")

with open(MODELNAME, 'rb') as f_in:
    model = pickle.load(f_in)


def predict_single(patient_dict: dict) -> float:
    # convert dict -> single-row DataFrame
    df = pd.DataFrame([patient_dict])

    # try to align columns with training order if available
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        df = df.loc[:, feature_names]

    prob = model.predict_proba(df)[0, 1]
    return float(prob)


@app.post("/predict")
def predict(patient: Patient) -> PredictDiabetes:
    prob = predict_single(patient.model_dump())

    return PredictDiabetes(
        diabetes_probability=prob,
        diabetes=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)