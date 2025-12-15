
"""
SQL utilities (SQLite for local prototyping):
- Master table creation
- Joins & window functions example
"""
from pathlib import Path
import sqlite3
import pandas as pd
from .config import PATHS
from .logging_utils import get_logger

logger = get_logger("sql_ops")

def get_conn(db_path: Path = PATHS.interim_dir / "hr.db") -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    return conn

def write_table(df: pd.DataFrame, table_name: str, conn: sqlite3.Connection) -> None:
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    logger.info(f"Wrote table {table_name} ({df.shape})")

def run_query(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info(f"Running query:\n{query}")
    return pd.read_sql_query(query, conn)

def example_window_query(conn: sqlite3.Connection) -> pd.DataFrame:
    q = """
    SELECT
      Department,
      EmployeeID,
      Salary,
      AVG(Salary) OVER (PARTITION BY Department) AS dept_avg_salary
    FROM employees;
    """
    return run_query(q, conn)
