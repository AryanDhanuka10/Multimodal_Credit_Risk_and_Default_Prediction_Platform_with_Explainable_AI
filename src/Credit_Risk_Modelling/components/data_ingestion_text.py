import logging
from pathlib import Path
from Credit_Risk_Modelling.utils.common import calculate_md5

class TextDataIngestion:
    def __init__(self, local_path: Path):
        self.local_path = local_path

    def ingest(self):
        logging.info("Validating text dataset")
        if not self.local_path.exists():
            raise FileNotFoundError("Text data not found")
        checksum = calculate_md5(self.local_path)
        logging.info(f"Text data checksum: {checksum}")
        return self.local_path
