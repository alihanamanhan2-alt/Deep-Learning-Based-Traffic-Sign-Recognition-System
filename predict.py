"""Load trained PyTorch checkpoints and predict GTSRB classes."""
from pathlib import Path
from typing import Iterable, Dict, List

import torch
from PIL import Image

from src.model import build_model
from src.preprocessing import build_transforms


class TrafficSignPredictor:
    """Inference wrapper that validates checkpoints and shares preprocessing."""

    def __init__(self, weights: Path, device: str = "cpu"):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.weights}. Train a model first."
            )
        self.device = torch.device(device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ))
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.model_name = str(checkpoint.get("model_name", "cnn"))
        self.image_size = int(checkpoint.get("image_size", 64))
        raw_names = checkpoint.get("class_names", {})
        self.class_names: Dict[int, str] = {}
        for i in range(43):
            if isinstance(raw_names, dict):
                if i in raw_names:
                    self.class_names[i] = str(raw_names[i])
                elif str(i) in raw_names:
                    self.class_names[i] = str(raw_names[str(i)])
                else:
                    self.class_names[i] = f"Class {i}"
            else:
                self.class_names[i] = str(raw_names[i]) if i < len(raw_names) else f"Class {i}"

        self.model = build_model(self.model_name, num_classes=43)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        _, self.transform = build_transforms(self.image_size)

    def predict_image(self, image: Image.Image, top_k: int = 5) -> Dict:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        rgb = image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        values, indices = probabilities.topk(min(top_k, 43))
        predictions = [
            {"class_id": int(index), "label": self.class_names[int(index)],
             "confidence": float(value)}
            for value, index in zip(values.cpu(), indices.cpu())
        ]
        return predictions[0] | {"top_predictions": predictions}

    def predict_batch(self, images: Iterable[Image.Image], top_k: int = 5) -> List[Dict]:
        return [self.predict_image(image, top_k=top_k) for image in images]
