import logging
from pathlib import Path

class DocumentDataIngestion:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def ingest(self):
        logging.info("Validating document image dataset")
        if not self.root_dir.exists():
            raise FileNotFoundError("Document image directory missing")
        total_images = len(list(self.root_dir.glob("**/*.jpg")))
        logging.info(f"Total document images found: {total_images}")
        return self.root_dir
