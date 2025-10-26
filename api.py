"""
Command to run it from terminal: uvicorn api:app --reload

===============================================================
FastAPI App for Adult Income Prediction
Uses the saved GradientBoosting Model (includes preprocessing)
===============================================================
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------------------
# Load the trained model
# --------------------------------------------------------------
# The model saved from Step 3 already contains both:
#  - Preprocessing (scaling, encoding, etc.)
#  - The trained GradientBoosting classifier
# So we can send raw inputs (after feature engineering) directly into it.
MODEL_PATH = Path(__file__).resolve().parent / "model" / "best_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "best_model.pkl not found. Export it from Step 3 (the GradientBoosting model with preprocessing included)."
    )

model = joblib.load(MODEL_PATH)  # GradientBoosting model with preprocessing

# --------------------------------------------------------------
# Initialize the FastAPI app
# --------------------------------------------------------------
app = FastAPI(
    title="Adult Income Prediction API",
    description="Predicts whether a person earns >$50K/year using the trained GradientBoosting model.",
    version="1.0"
)

# --------------------------------------------------------------
# Define input data structure (matches original dataset fields)
# --------------------------------------------------------------
class Person(BaseModel):
    age: int
    workclass: str
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


# --------------------------------------------------------------
# Health check endpoint
# --------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Adult Income Prediction API is running."}


# --------------------------------------------------------------
# Prediction endpoint
# --------------------------------------------------------------
@app.post("/predict")
def predict(person: Person):
    """
    API prediction workflow:
      1 Convert the input into a DataFrame.
      2 Rename columns to match the ones used during training.
      3 Recreate the deterministic engineered features (from Step 2).
      4 Feed the data directly into the loaded model (it already preprocesses internally).
      5 Return the prediction and probability.
    """

    # Step 1: Convert input to DataFrame
    df = pd.DataFrame([person.model_dump()])

    # Step 2: Rename columns to match training format
    df.rename(
        columns={
            "education_num": "education-num",
            "marital_status": "marital-status",
            "hours_per_week": "hours-per-week",
            "native_country": "native-country",
            "capital_gain": "capital-gain",
            "capital_loss": "capital-loss",
        },
        inplace=True,
    )

    # Step 3: Deterministic feature engineering (like Step 2)
    df["is_senior"] = (df["age"] >= 60).astype(int)
    df["capital-gain_log"] = np.log1p(df["capital-gain"])
    df["capital-loss_log"] = np.log1p(df["capital-loss"])
    df["education_per_age"] = (df["education-num"] / df["age"].replace(0, np.nan)).fillna(0)
    df["work_gain_ratio"] = df["hours-per-week"] * (df["capital-gain_log"] + 1)
    df["is_married"] = df["marital-status"].apply(lambda x: 1 if "Married" in str(x) else 0)
    df["gender_role"] = df["sex"].astype(str) + "_" + df["relationship"].astype(str)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["<25", "25–35", "35–45", "45–55", "55–65", "65+"],
        right=False,
    )

    # Step 4: Predict using the trained model
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else None

    # Step 5: Return result
    return {
        "model_used": "GradientBoosting (with preprocessing)",
        "prediction": pred,          # 0 = <=50K, 1 = >50K
        "probability(>50K)": prob,   # Probability of earning >50K
    }
