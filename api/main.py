from pathlib import Path

import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

app = FastAPI(
    title="API - Prédiction du prix automobile"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_model.joblib"
)

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "API de prédiction du prix automobile",
        "model": "Lasso - modèle champion"
    }




class VehicleInput(BaseModel):
    YEAR: int
    KM: float
    BODYDOORS: int
    PASSENGERS: int
    MAKE: str
    MODEL: str
    BODYDESCF: str
    DRIVETRAIN: str
    FUEL: str
    TRANSDESCF: str
    COLORF: str


@app.post("/predict")
def predict(vehicle: VehicleInput):

    input_data = pd.DataFrame(
        [vehicle.model_dump()]
    )

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }