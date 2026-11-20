"""Compare evaluated PyTorch model artifacts and generate no-fake evidence tables."""
import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR


def compare_models(output_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for model_dir in sorted(output_dir.glob("*")):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(
            "No metrics.json files were found. Run python evaluate.py for both trained models first."
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "model_comparison.csv", index=False)

    columns = ["model", "accuracy", "precision_weighted", "recall_weighted",
               "f1_weighted", "test_loss", "test_samples"]
    markdown = table[[column for column in columns if column in table.columns]].to_markdown(index=False)
    (output_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return table


if __name__ == "__main__":
    compare_models()
