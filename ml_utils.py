from typing import List
from sklearn.pipeline import Pipeline
import pandas as pd
from .data_cleaning import build_preprocessor, split_features_target

def train_performance_model(df: pd.DataFrame, cfg) -> Pipeline:
    required_cols = cfg.cat_cols + cfg.num_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in DataFrame: {missing_cols}")
        raise KeyError(f"Missing columns in DataFrame: {missing_cols}")
    X = df[required_cols]
    y = df[cfg.target_col]
    preprocessor = build_preprocessor(df, cfg.cat_cols, cfg.num_cols)
    from sklearn.linear_model import LinearRegression
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ], memory=None)
    model.fit(X, y)
    return model

from typing import List
from sklearn.pipeline import Pipeline
import pandas as pd
from .data_cleaning import build_preprocessor, split_features_target

class AttritionConfig:
    def __init__(self, cat_cols: List[str], num_cols: List[str], target_col: str):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col

def train_attrition_model(df: pd.DataFrame, cfg: AttritionConfig) -> Pipeline:
    X, y = split_features_target(df, cfg.target_col)
    preprocessor = build_preprocessor(df, cfg.cat_cols, cfg.num_cols)
    from sklearn.linear_model import LogisticRegression
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ], memory=None)
    model.fit(X, y)
    return model
