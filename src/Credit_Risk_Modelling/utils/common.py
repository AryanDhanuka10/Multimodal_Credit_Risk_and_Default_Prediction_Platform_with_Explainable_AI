import os
import urllib.request
import hashlib
from pathlib import Path
import logging

def download_file(url: str, dest: Path):
    os.makedirs(dest.parent, exist_ok=True)
    logging.info(f"Downloading data from {url}")
    urllib.request.urlretrieve(url, dest)

def calculate_md5(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
