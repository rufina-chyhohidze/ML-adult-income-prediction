# ============================
# STEP 2: Reproducible Pipeline
# ============================
# Why scikit-learn Pipelines?
# - Every transform (imputation, encoding, scaling, feature selection) is FIT only on training folds.
# - This prevents data leakage and makes the workflow reproducible & deployable (project requirement #2).  # see brief

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 0) Data loading + EDA cleaning (programmatic)
# -----------------------------
def load_and_basic_clean(train_path="data/adult.data", test_path="data/adult.test"):
    column_names = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df_train = pd.read_csv(train_path, names=column_names)
    df_test  = pd.read_csv(test_path, skiprows=1, names=column_names)

    # Replace '?' -> 'Unknown' for categorical columns we inspected in EDA
    for col in ["workclass", "occupation", "native-country"]:
        df_train[col] = df_train[col].replace("?", "Unknown")
        df_test[col]  = df_test[col].replace("?", "Unknown")

    # Trim whitespace across object columns
    for df in (df_train, df_test):
        obj_cols = df.select_dtypes(include="object").columns
        for c in obj_cols:
            df[c] = df[c].str.strip()

    # Normalize target labels (remove trailing dot in test)
    df_train["income"] = df_train["income"].str.replace(".", "", regex=False)
    df_test["income"]  = df_test["income"].str.replace(".", "", regex=False)

    # Drop columns per EDA decisions
    df_train = df_train.drop(columns=["fnlwgt", "education"])
    df_test  = df_test.drop(columns=["fnlwgt", "education"])

    return df_train, df_test

df_train_data, df_test_data = load_and_basic_clean()

# Quick sanity check of columns
print("Columns:", df_train_data.columns.tolist())
print(df_train_data.head(2))

# -----------------------------
# 1) Feature/target split
# -----------------------------
TARGET = "income"
X = df_train_data.drop(columns=[TARGET])
y = (df_train_data[TARGET] == ">50K").astype(int)   # encode as 0/1 for sklearn models

# We'll keep df_test_data aside as a true holdout for Step 4 (evaluation).
X_test_holdout = df_test_data.drop(columns=[TARGET])
y_test_holdout = (df_test_data[TARGET] == ">50K").astype(int)

# -----------------------------
# 2) Column groups for preprocessing
# -----------------------------
# Numeric columns
numeric_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()

# In our EDA, we noted heavy right skew in capital-gain/loss, so we’ll transform those.
skewed_numeric = [c for c in ["capital-gain", "capital-loss"] if c in numeric_cols_all]
base_numeric   = [c for c in numeric_cols_all if c not in skewed_numeric]

# Categorical columns (everything else that’s object)
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Base numeric:", base_numeric)
print("Skewed numeric:", skewed_numeric)
print("Categorical:", categorical_cols)

# -----------------------------
# 3) Preprocessing blocks (no leakage)
# -----------------------------
# Why these choices?
# - SimpleImputer: robust against any accidental NaNs (future-proofing).
# - FunctionTransformer(log1p): reduces skew ONLY for capital-gain/loss (helps linear models).
# - StandardScaler: helps linear models; tree-based models ignore it but harmless.
# - OneHotEncoder(handle_unknown='ignore', min_frequency=0.01): handles unseen categories + auto-groups rare ones.
#   This is superior to manual grouping and prevents exploding feature space on ultra-rare categories.

numeric_base_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_skewed_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),   # falls back to most common; we also have 'Unknown'
    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False))
])

# ColumnTransformer stitches everything together and keeps column order stable
preprocess = ColumnTransformer(
    transformers=[
        ("num_base", numeric_base_pipeline, base_numeric),
        ("num_skewed", numeric_skewed_pipeline, skewed_numeric),
        ("cat", categorical_pipeline, categorical_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# -----------------------------
# 4) (Optional) Feature selection inside the pipeline
# -----------------------------
# Why: after OHE, we can end up with many columns. Mutual information is non-linear, robust for classification.
# Keep this OFF by default. You can set k to an int (e.g., 60) to enable.
K_BEST = None  # set to an integer to enable, e.g., 60

if K_BEST is not None:
    feature_selection = SelectKBest(score_func=mutual_info_classif, k=K_BEST)
else:
    feature_selection = "passthrough"

# -----------------------------
# 5) Build a modeling-ready pipeline (estimator here is just a placeholder)
# -----------------------------
# We attach a simple estimator ONLY to smoke-test the pipeline.
# In Step 3 you'll swap/compare multiple models (LogReg, RF, XGB/GBDT) with hyperparameter tuning.
pipe_logreg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("feat_sel", feature_selection),
    ("clf", LogisticRegression(max_iter=1000, n_jobs=None, class_weight=None))
])

# -----------------------------
# 6) Quick smoke test (train/val split) — validates the pipeline works end-to-end
# -----------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe_logreg.fit(X_tr, y_tr)
val_score = pipe_logreg.score(X_val, y_val)
print(f"[Smoke test] LogisticRegression validation accuracy: {val_score:.3f}")

# Note:
# - We deliberately avoid any manual preprocessing outside the pipeline.
# - For class imbalance, in Step 3 you can try class_weight='balanced' (LogReg) or tuned thresholds.
# - For tree models (RF/GBDT), StandardScaler is inert; you can reuse the SAME preprocess safely.
