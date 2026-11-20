"""Evaluate a trained GTSRB model and create real result artifacts."""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import MODEL_DIR, RESULTS_DIR
from src.dataset import GTSRB_CLASS_NAMES, create_dataloaders
from src.model import build_model
from src.metrics import save_confusion_matrix, save_training_plots
from src.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB model")
    parser.add_argument("--model", choices=["cnn", "resnet18"], required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/GTSRB"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    return y_true, y_pred, total_loss / len(loader.dataset)


def main():
    args = parse_args()
    weights = args.weights or (MODEL_DIR / f"{args.model}_best.pth")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights}. Train the model with python train.py --model {args.model} first."
        )

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    image_size = int(checkpoint.get("image_size", 64))
    class_names_map = checkpoint.get("class_names")
    if class_names_map:
        class_names = [class_names_map[str(i)] if str(i) in class_names_map else class_names_map[i]
                       for i in range(43)]
    else:
        class_names = GTSRB_CLASS_NAMES

    _, _, test_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    device = get_device(args.device)
    model = build_model(args.model, num_classes=43)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    y_true, y_pred, test_loss = evaluate(model, test_loader, device)
    report = classification_report(
        y_true, y_pred, labels=list(range(43)), target_names=class_names,
        zero_division=0, output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(43)))

    output_dir = RESULTS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png")

    history = checkpoint.get("history", {})
    if history:
        save_training_plots(history, output_dir)

    aggregate = report["weighted avg"]
    summary = {
        "model": args.model,
        "test_loss": test_loss,
        "accuracy": report["accuracy"],
        "precision_weighted": aggregate["precision"],
        "recall_weighted": aggregate["recall"],
        "f1_weighted": aggregate["f1-score"],
        "test_samples": len(y_true),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
