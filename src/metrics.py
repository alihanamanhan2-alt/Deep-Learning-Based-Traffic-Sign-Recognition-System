"""Metric computation and plot generation for model evaluation."""
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
import seaborn as sns


def calculate_metrics(y_true, y_pred, class_names=None) -> Dict:
    """Return aggregate metrics and the full per-class report."""
    labels = list(range(len(class_names))) if class_names else None
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=class_names,
            zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_training_plots(history: Dict, output_dir: Path) -> None:
    """Save loss and accuracy curves from an actual training history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, title, filename in [
        ("loss", "Training and validation loss", "loss.png"),
        ("accuracy", "Training and validation accuracy", "accuracy.png"),
    ]:
        train_values, val_values = history.get(key, []), history.get(f"val_{key}", [])
        if not train_values:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_values, label="Train")
        if val_values:
            ax.plot(val_values, label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=key.capitalize())
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def save_confusion_matrix(matrix, class_names, output_path: Path) -> None:
    """Save confusion matrix generated from test predictions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(np.asarray(matrix), cmap="Blues", ax=ax, cbar=True)
    ax.set(title="GTSRB confusion matrix", xlabel="Predicted class", ylabel="True class")
    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
