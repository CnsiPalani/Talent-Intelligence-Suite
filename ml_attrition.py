
# src/ml_attrition.py (add a DB-based loader)
from .db_utils import read_sql_df
from .ml_utils import AttritionConfig, train_attrition_model
from sklearn.pipeline import Pipeline

def train_attrition_model_from_db(cfg: AttritionConfig,query: str) -> Pipeline:
    df = read_sql_df(query)
    # Map to expected columns
    df["Attrition"] = df["Attrition"].astype(int)  # ensure 0/1
    return train_attrition_model(df, cfg)

cfg = AttritionConfig(
    cat_cols=["Department", "JobRole", "status", "location_code"],
    num_cols=["YearsAtCompany", "MonthlyIncome"],
    target_col="Attrition"
)