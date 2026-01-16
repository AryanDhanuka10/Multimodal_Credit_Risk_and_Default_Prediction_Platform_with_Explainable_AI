import cv2
import numpy as np
from pathlib import Path
import random

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = Path("artifacts/data_ingestion/documents/images/base")
LOW_RISK_DIR = Path("artifacts/data_ingestion/documents/images/low_risk")
HIGH_RISK_DIR = Path("artifacts/data_ingestion/documents/images/high_risk")

LOW_RISK_DIR.mkdir(parents=True, exist_ok=True)
HIGH_RISK_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# IMAGE TRANSFORMATIONS
# =========================
def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def random_crop(img):
    h, w, _ = img.shape
    crop_h, crop_w = int(h * 0.8), int(w * 0.8)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return img[y:y + crop_h, x:x + crop_w]

def adjust_brightness(img):
    factor = random.uniform(0.5, 1.5)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

# =========================
# MAIN GENERATION LOGIC
# =========================
def main():
    print("[INFO] Starting synthetic document generation")

    base_images = list(BASE_DIR.glob("*"))
    print(f"[INFO] Found {len(base_images)} base images")

    if len(base_images) == 0:
        print("[ERROR] Base directory is empty. Add at least one image.")
        return

    valid_ext = [".jpg", ".jpeg", ".png"]
    image_count = 0

    for img_path in base_images:
        if img_path.suffix.lower() not in valid_ext:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Could not read {img_path.name}")
            continue

        print(f"[INFO] Processing {img_path.name}")

        # -----------------
        # LOW RISK (CLEAN)
        # -----------------
        low_risk_path = LOW_RISK_DIR / img_path.name
        cv2.imwrite(str(low_risk_path), img)
        image_count += 1

        # -----------------
        # HIGH RISK (CORRUPTED)
        # -----------------
        corruptions = [
            add_noise,
            blur,
            random_crop,
            adjust_brightness
        ]

        for i in range(3):
            transform = random.choice(corruptions)
            corrupted = transform(img)

            out_name = f"{img_path.stem}_risk_{i}.jpg"
            out_path = HIGH_RISK_DIR / out_name
            cv2.imwrite(str(out_path), corrupted)
            image_count += 1

    print(f"[SUCCESS] Generated {image_count} synthetic document images")
    print(f"[INFO] Low-risk images: {len(list(LOW_RISK_DIR.glob('*')))}")
    print(f"[INFO] High-risk images: {len(list(HIGH_RISK_DIR.glob('*')))}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
