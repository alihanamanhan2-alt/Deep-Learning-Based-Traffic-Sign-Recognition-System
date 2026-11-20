"""Train a custom CNN or ResNet18 on GTSRB."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (BATCH_SIZE, DATA_DIR, EPOCHS, IMAGE_SIZE, MODEL_DIR,
                    RESULTS_DIR, SEED)
from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import get_device, save_checkpoint, seed_everything


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Run one training or validation epoch and return loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, disable=not training):
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GTSRB PyTorch model")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true",
                        help="Initialize ResNet18 with torchvision ImageNet weights")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    seed_everything(SEED)
    device = get_device(args.device)

    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=SEED,
    )
    model = build_model(args.model, num_classes=len(class_names), pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_accuracy = -1.0
    checkpoint_path = MODEL_DIR / f"{args.model}_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device=device,
        )
        scheduler.step(val_accuracy)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path, model, args.model, class_names,
                args.image_size, history=history,
            )

    (RESULTS_DIR / args.model).mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / args.model / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Weights: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
