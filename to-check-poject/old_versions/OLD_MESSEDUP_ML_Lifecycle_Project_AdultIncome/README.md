# Machine Learning Lifecycle Project — Adult Income

This repo implements the full ML lifecycle per your course brief, with **detailed explanations** at each step.

- **Dataset**: Adult Income (files in `data/`, from your upload)
- **Task**: Binary classification — predict if `income` is `>50K`

## What's inside
- `notebooks/01_ML_Lifecycle_AdultIncome.ipynb` — a single, comprehensive notebook covering:
  1) Problem framing & data understanding
  2) EDA with commentary
  3) Leakage-safe preprocessing & feature engineering
  4) Model selection + tuning (multiple models)
  5) Validation & metrics
  6) Error analysis & interpretability
  7) Persisting the final model
  8) (Bonus) Serving via FastAPI
- `src/pipeline.py` — reusable preprocessing pipeline
- `api/app.py` — (bonus) FastAPI app to serve the trained model
- `reports/figures/` — where figures are saved when you run the notebook
- `models/` — where the trained model is saved when you run the notebook
- `rubric/` — your rubric file (copied)
- `Project Description.pdf` — your project brief (copied)
- `AI_tools_usage.txt` — required log
- `requirements.txt` — install these locally

## How to run
```bash
pip install -r requirements.txt
# open the notebook and run all cells
jupyter lab notebooks/OLD_01_ML_Lifecycle_AdultIncome.ipynb
# (bonus) after training finishes:
uvicorn api.app:app --reload
```
