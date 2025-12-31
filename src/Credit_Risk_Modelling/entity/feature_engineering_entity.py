from dataclasses import dataclass
from pathlib import Path

@dataclass
class TimeSeriesFeatureConfig:
    data_path: Path
    output_path: Path
    window_size: int
