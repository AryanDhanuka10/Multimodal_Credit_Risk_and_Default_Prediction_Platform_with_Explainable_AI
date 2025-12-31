from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataValidationConfig:
    required_columns: list
    data_path: Path
