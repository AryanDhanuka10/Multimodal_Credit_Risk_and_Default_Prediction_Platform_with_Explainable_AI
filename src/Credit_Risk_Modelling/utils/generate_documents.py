import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random

OUTPUT_DIR = "artifacts/data_ingestion/documents/images"
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
SUSPICIOUS_DIR = os.path.join(OUTPUT_DIR, "suspicious")

NUM_IMAGES = 300


def generate_document(text, noisy=False):
    img = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

    y = 50
    for line in text:
        draw.text((50, y), line, font=font, fill="black")
        y += 40

    img_np = np.array(img)

    if noisy:
        noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

    return img_np


def generate_dataset():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(SUSPICIOUS_DIR, exist_ok=True)

    for i in range(NUM_IMAGES):
        salary = random.randint(20000, 120000)
        doc_text = [
            "Salary Statement",
            f"Employee ID: {random.randint(10000,99999)}",
            f"Monthly Salary: ₹{salary}",
            "Company: XYZ Pvt Ltd",
            "Authorized Signature"
        ]

        clean_img = generate_document(doc_text, noisy=False)
        suspicious_img = generate_document(doc_text, noisy=True)

        cv2.imwrite(f"{CLEAN_DIR}/doc_clean_{i}.png", clean_img)
        cv2.imwrite(f"{SUSPICIOUS_DIR}/doc_susp_{i}.png", suspicious_img)


if __name__ == "__main__":
    generate_dataset()
    print("Synthetic document dataset generated.")
