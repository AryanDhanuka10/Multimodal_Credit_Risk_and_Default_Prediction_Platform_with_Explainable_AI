import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import joblib


class DocumentFeatureEngineering:
    def __init__(self, image_dir: Path, output_dir: Path):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pretrained backbone (industry standard)
        backbone = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def extract_embeddings(self):
        embeddings = []
        labels = []

        for label_dir in ["low_risk", "high_risk"]:
            class_dir = self.image_dir / label_dir
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpg"):
                image = Image.open(img_path).convert("RGB")
                tensor = self.transform(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    emb = self.model(tensor).squeeze().cpu().numpy()

                embeddings.append(emb)
                labels.append(1 if label_dir == "high_risk" else 0)

        embeddings = np.array(embeddings)
        labels = np.array(labels)

        joblib.dump(
            {"embeddings": embeddings, "labels": labels},
            self.output_dir / "document_embeddings.pkl"
        )

        return embeddings, labels
