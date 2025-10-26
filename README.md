# Data & AI 5 - Adult Income Prediction (ML Lifecycle)

## Team 28
**Anir, Rufina, Daria**

Artificial Intelligence - Machine Learning Lifecycle Project

---

## Goal
Build a reliable classifier to predict whether an individual’s annual income exceeds **$50,000** using the **Adult (Census Income)** dataset, while **demonstrating the full ML lifecycle**: EDA -> data cleaning & feature engineering -> model selection & validation -> final evaluation -> simple API deployment.

---

## Dataset
- Source: Adult/Census Income (15 columns: 14 features + target `income`).
- Target: `income` (binary) - mapped to **0 = <=50K**, **1 = >50K**.
- Class balance (approx.): **~76% <=50K**, **~24% >50K** (imbalance considered during training).
- Notable issues handled: missing categorical values (`workclass`, `occupation`, `native-country`), whitespace/format inconsistencies (e.g., `>50K.` in test labels), and a non‑predictive ID-like column (`fnlwgt`).

---

## Project Structure (notebooks)
1. **Step 1 - Data Understanding & EDA**  
   Exploratory analysis, distributions, correlations, target balance, and data issues to address.
2. **Step 2 - Implementation & Data Pipeline (Feature Engineering only)**  
   - **Train‑driven cleaning** applied to *both* splits (stats from train, applied to test).  
   - **Deterministic feature engineering** applied to both splits:  
     - `is_senior` (age ≥ 60)  
     - `capital-gain_log`, `capital-loss_log` (log1p)  
     - `education_per_age = education-num / age` (guarding against /0)  
     - `work_gain_ratio = hours-per-week * (capital-gain_log + 1)`  
     - `is_married` (Marital status contains “Married”)  
     - `gender_role = sex + '_' + relationship`  
     - `age_group` (binned ages)  
   - **Artifacts saved** (for modeling):  
     - `artifacts/adult_train_clean_fe.csv`  
     - `artifacts/adult_test_clean_fe.csv`  
     - `artifacts/feature_lists.joblib` (numeric/categorical column lists)  
     - `artifacts/preprocessor_def.joblib` (unfitted structure for reference; the final model contains its own preprocessing)
3. **Step 3 - Model Selection & Evaluation**  
   - **Validation:** Stratified train/validation split; **StratifiedKFold (k=5)** for model comparison.  
   - **Preprocessing inside the model:**  
     `ColumnTransformer` -> numeric (`PowerTransformer` -> `StandardScaler`) + categorical (`OneHotEncoder(handle_unknown='ignore')`).  
   - **Class imbalance:** training optionally undersampled **train only** (never validation/test) to compare effects; final metrics reported on the **original (imbalanced) validation/test**.  
   - **Models compared:** Logistic Regression, SVC, Random Forest, **Gradient Boosting** (others explored as baselines).  
   - **Model choice:** **Gradient Boosting** selected based on ROC‑AUC and stable validation behavior.

---

## Results (summary)
- **Cross‑Validation (mean ROC‑AUC):** ~**0.922** (Gradient Boosting).  
- **Hold‑out Validation ROC‑AUC:** **0.9376**.  
- Additional indicators (precision/recall) show improved minority‑class recognition versus linear baselines, with acceptable trade‑off in interpretability.

**Conclusion.** Gradient Boosting delivered the best overall performance and generalization for this problem, handling mixed numeric/categorical signals well after our engineered features and robust preprocessing. For production, we exported the **full trained model (preprocessing + classifier)** as a single artifact.

---

## How to Reproduce

### 1) Environment
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

### 2) Run notebooks
Execute in order (ensure `./artifacts` and `./data` exist):
1. `1_Data_Understanding_and_Exploration.ipynb`
2. `2_Implementation_and_Data_Pipeline.ipynb` (feature engineering only)  
   -> produces cleaned + FE data and helper artifacts in `./artifacts/`
3. `3_4_Model_Selection_Evaluation.ipynb`  
   -> performs CV, selects best model, and saves `model/best_model.pkl`

### 3) Artifacts produced
- `artifacts/adult_train_clean_fe.csv` (cleaned + engineered train)
- `artifacts/adult_test_clean_fe.csv` (cleaned + engineered test)
- `artifacts/feature_lists.joblib` (column lists)
- `model/best_model.pkl` (**final model including preprocessing**)

---

## API Deployment (FastAPI)

### Start the API
From the project root, with `model/best_model.pkl` present:
```bash
docker compose up
```
- Use the test_api.http file to test it
- The api will be available by default at http://localhost:8000/docs
- **Endpoint:**  `POST /predict`
- **Input JSON schema (raw Adult fields):**
  ```json
  {
    "age": 39,
    "workclass": "Private",
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
  }
  ```
- The server **recreates the engineered features** (deterministic) and calls the **saved model**, which applies its own preprocessing and returns:
  ```json
  {
    "model_used": "GradientBoosting (with preprocessing)",
    "prediction": 0,
    "probability(>50K)": 0.13514112523261307
  }
  ```

---

## Design Choices & Justification
- **Leakage prevention:** All imputations/statistics learned **from train only** and then applied to test; validation/test are never balanced or fitted on.  
- **Feature engineering:** The same fixed formulas are used in both notebooks and the API to ensure consistent results.  
- **Model selection:** Compared multiple families, Gradient Boosting was most robust on ROC‑AUC with balanced precision/recall on minority class.  
- **Single “model” export:** Packaging preprocessing + classifier ensures consistent inference and simpler API code.

---

Course: **Data & AI 5 / Artificial Intelligence - Machine Learning Lifecycle Project (2025)**.  
Teachers: Alexander Michielsen, Jan Van Sas.
