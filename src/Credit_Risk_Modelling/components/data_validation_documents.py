import logging
from pathlib import Path

class DocumentDataValidation:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def validate(self):
        logging.info("Validating document image data")

        if not self.data_dir.exists():
            raise FileNotFoundError("Document directory does not exist")

        image_files = list(self.data_dir.glob("**/*.jpg")) + list(self.data_dir.glob("**/*.png"))

        if len(image_files) < 100:
            raise ValueError("Too few document images found")

        logging.info(f"Validated {len(image_files)} document images")
        return True
