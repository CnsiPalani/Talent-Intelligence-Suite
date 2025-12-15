
from src.db_utils import read_sql_df
PERF_SQL = """
SELECT e.employee_id, d.name AS Department, jr.title AS JobRole,
       e.compensation_base AS MonthlyIncome,
       TIMESTAMPDIFF(YEAR, e.hire_date, CURRENT_DATE) AS YearsAtCompany,
       COALESCE(pr.overall_rating, 3) AS PerformanceScore
FROM employee e
LEFT JOIN department d ON d.department_id = e.department_id
LEFT JOIN job_role jr ON jr.job_role_id = e.job_role_id
LEFT JOIN (
  SELECT employee_id, MAX(period_end) AS latest_end
  FROM performance_review
  GROUP BY employee_id
) latest ON latest.employee_id = e.employee_id
LEFT JOIN performance_review pr
  ON pr.employee_id = latest.employee_id AND pr.period_end = latest.latest_end;
"""

from src.perf_config import PerfConfig
from src.ml_utils import train_performance_model

def train_performance_model_from_db(cfg: PerfConfig):
    df = read_sql_df(PERF_SQL)
    return train_performance_model(df, cfg)
