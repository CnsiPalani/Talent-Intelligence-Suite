
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"
    
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")

@dataclass
class Settings:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

PATHS = Paths()
SETTINGS = Settings()
