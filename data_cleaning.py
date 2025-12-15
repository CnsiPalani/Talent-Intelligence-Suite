
"""
Data cleaning and feature engineering:
- Handling missing values
- One-hot encoding
- Scaling numeric features
- Text cleaning (tokenization, lemmatization placeholders)
"""
from typing import Tuple, List
import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from .logging_utils import get_logger

logger = get_logger("data_cleaning")

def basic_text_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_preprocessor(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Split data into X({X.shape}) and y({y.shape})")
    return X, y
