
# src/db_utils.py
from typing import Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .db_config import connection_url, DB
from .logging_utils import get_logger

logger = get_logger("db_utils")

_engine: Optional[Engine] = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = connection_url()
        logger.info(f"Creating SQLAlchemy engine to {DB.host}:{DB.port}/{DB.database} via {DB.driver}")
        _engine = create_engine(
            url,
            pool_size=DB.pool_size,
            pool_recycle=DB.pool_recycle,
            future=True,
        )
    return _engine

def read_sql_df(sql: str, params: Optional[Dict[str, Any]] = None, chunksize: Optional[int] = None) -> pd.DataFrame:
    """Read SQL into a DataFrame; optionally stream in chunks and concat."""
    eng = get_engine()
    if chunksize:
        logger.info(f"Streaming query in chunksize={chunksize}")
        frames = []
        with eng.connect() as conn:
            for chunk in pd.read_sql(text(sql), conn, params=params, chunksize=chunksize):
                frames.append(chunk)
        return pd.concat(frames, ignore_index=True)
    else:
        with eng.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

def execute(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    """Execute DML/DDL safely."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})
