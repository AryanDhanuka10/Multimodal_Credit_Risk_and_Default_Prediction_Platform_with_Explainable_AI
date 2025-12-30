from box import ConfigBox
from pathlib import Path
import yaml

class ConfigurationManager:
    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = ConfigBox(yaml.safe_load(f))

    def get_data_ingestion_config(self):
        return self.config.data_ingestion
