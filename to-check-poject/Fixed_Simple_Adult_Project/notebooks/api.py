# ==============================================
# FastAPI App for Adult Income Prediction
# ==============================================

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Load trained model (the full pipeline)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "final_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Adult Income Prediction API",
    description="Predict whether a person earns >50K/year based on demographic data.",
    version="1.0"
)

# Define the input schema (match your model features)
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
def home():
    return {"message": "Adult Income Prediction API is running 🚀"}

@app.post("/predict")
def predict(person: Person):
    # Convert input JSON -> pandas DataFrame
    data = pd.DataFrame([person.dict()])

    # Rename to match model's training feature names
    rename_map = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss"
    }
    data = data.rename(columns=rename_map)

    # Predict
    pred = model.predict(data)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = round(model.predict_proba(data)[0][1], 4)

    return {
        "prediction": pred,
        "probability(>50K)": prob
    }
