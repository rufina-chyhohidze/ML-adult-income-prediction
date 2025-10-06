from fastapi import FastAPI
from typing import List, Any, Dict
import pandas as pd
import joblib, os

app = FastAPI(title="Adult Income Model API", version="1.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.joblib")
clf = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "education-num": "education-num",
        "capital-gain": "capital-gain",
        "capital-loss": "capital-loss",
        "hours-per-week": "hours-per-week",
        "marital-status": "marital-status",
        "native-country": "native-country",
        "education_num": "education-num",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "marital_status": "marital-status",
        "native_country": "native-country",
    }
    out = {}
    for k, v in d.items():
        out[mapping.get(k, k)] = v
    return out

@app.post("/predict")
def predict(rows: List[dict]):
    if clf is None:
        return {"error": "Model not found. Train and save 'models/final_model.joblib' first."}
    normed = [normalize_keys(r) for r in rows]
    df = pd.DataFrame(normed)
    preds = clf.predict(df)
    proba = clf.predict_proba(df)[:, 1] if hasattr(clf, "predict_proba") else [None] * len(preds)
    return [{"prediction": p, "probability_gt50k": float(prob) if prob is not None else None}
            for p, prob in zip(preds, proba)]
