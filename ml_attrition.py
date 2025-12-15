
# src/ml_attrition.py (add a DB-based loader)
from .db_utils import read_sql_df
from .ml_utils import AttritionConfig, train_attrition_model
from sklearn.pipeline import Pipeline
ATTRITION_SQL = """
SELECT e.employee_id,
       d.name AS Department, jr.title AS JobRole, e.status,
       TIMESTAMPDIFF(YEAR, e.hire_date, CURRENT_DATE) AS YearsAtCompany,
       e.compensation_base AS MonthlyIncome,
       e.location_code,
       e.attrition_flag AS Attrition
FROM employee e
LEFT JOIN department d ON d.department_id = e.department_id
LEFT JOIN job_role jr ON jr.job_role_id = e.job_role_id
WHERE e.status IN ('Active','OnLeave','Terminated');
"""

def train_attrition_model_from_db(cfg: AttritionConfig) -> Pipeline:
    df = read_sql_df(ATTRITION_SQL)
    # Map to expected columns
    df["Attrition"] = df["Attrition"].astype(int)  # ensure 0/1
    return train_attrition_model(df, cfg)

cfg = AttritionConfig(
    cat_cols=["Department", "JobRole", "status", "location_code"],
    num_cols=["YearsAtCompany", "MonthlyIncome"],
    target_col="Attrition"
)