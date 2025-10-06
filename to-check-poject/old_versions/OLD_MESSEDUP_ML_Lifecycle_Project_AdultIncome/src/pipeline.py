from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CATEGORICAL_GUESS = ["workclass","education","marital-status","occupation",
                     "relationship","race","sex","native-country"]
NUMERIC_GUESS = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

TARGET_NAME = "income"

def build_preprocess_pipeline(X: pd.DataFrame,
                              categorical: List[str] | None = None,
                              numeric: List[str] | None = None) -> Tuple[Pipeline, List[str], List[str]]:
    if categorical is None:
        categorical = [c for c in X.columns if c in CATEGORICAL_GUESS]
    if numeric is None:
        numeric = [c for c in X.columns if c in NUMERIC_GUESS]

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )
    pipe = Pipeline(steps=[("preprocess", pre)])
    return pipe, categorical, numeric
