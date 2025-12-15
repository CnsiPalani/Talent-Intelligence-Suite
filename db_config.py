
# src/db_config.py
import os
from dataclasses import dataclass

@dataclass
class DBSettings:
    # Read from environment variables to avoid hardcoding secrets
    user: str = os.getenv("HRTECH_DB_USER", "root")
    password: str = os.getenv("HRTECH_DB_PASSWORD", "password")
    host: str = os.getenv("HRTECH_DB_HOST", "127.0.0.1")
    port: str = os.getenv("HRTECH_DB_PORT", "3306")
    database: str = os.getenv("HRTECH_DB_NAME", "hrtech")
    driver: str = os.getenv("HRTECH_DB_DRIVER", "mysql+mysqldb")  # or "mysql+pymysql"
    pool_size: int = int(os.getenv("HRTECH_DB_POOL_SIZE", "5"))
    pool_recycle: int = int(os.getenv("HRTECH_DB_POOL_RECYCLE", "1800"))  # seconds

DB = DBSettings()

def connection_url() -> str:
    return f"{DB.driver}://{DB.user}:{DB.password}@{DB.host}:{DB.port}/{DB.database}?charset=utf8mb4"
