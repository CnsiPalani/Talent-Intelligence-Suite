
from .db_utils import read_sql_df
WORKLOAD_SQL = """
SELECT log_ts, work_items_count
FROM workload_log
WHERE log_ts BETWEEN :start_dt AND :end_dt
ORDER BY log_ts;
"""

def load_workload_series(start_dt: str, end_dt: str):
    df = read_sql_df(WORKLOAD_SQL, params={"start_dt": start_dt, "end_dt": end_dt})
    s = df.set_index("log_ts")["work_items_count"].asfreq("D").fillna(method="ffill")
    return s
