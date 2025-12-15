def eda_overview(df, report_name: str = "eda_employee"):
    # Dummy implementation: replace with actual EDA logic
    print(f"EDA Overview for {report_name}: {len(df)} rows")
    # Add more summary/statistics/plots as needed
    return {
        "report_name": report_name,
        "row_count": len(df)
    }

# src/eda.py (new function)
from .db_utils import read_sql_df
from .logging_utils import get_logger
logger = get_logger("eda")

HR_EMP_QUERY = """
SELECT e.employee_id, d.name AS Department, jr.title AS JobRole, e.status,
       TIMESTAMPDIFF(YEAR, e.hire_date, CURRENT_DATE) AS YearsAtCompany,
       e.compensation_base AS MonthlyIncome, e.attrition_flag AS Attrition
FROM employee e
LEFT JOIN department d ON d.department_id = e.department_id
LEFT JOIN job_role jr ON jr.job_role_id = e.job_role_id;
"""

def eda_overview_db():
    logger.info("Loaded employee data for EDA")
    df = read_sql_df(HR_EMP_QUERY, chunksize=None)
    return df