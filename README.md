# Traffic Sign Recognition — GTSRB

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download GTSRB and place the files under:

```text
data/raw/GTSRB/
├── Train/
│   ├── 0/
│   └── .../42/
├── Test/
├── Test.csv
└── Meta/
```

Official 43-class GTSRB releases use image folders named `0`–`42`. Some Kaggle archives also contain `Train.csv`; this project constructs training labels from the `Train/<class_id>/` folders and reads test labels from `Test.csv`.

## Train

```bash
python train.py --model cnn --epochs 20
python train.py --model resnet18 --epochs 20 --pretrained
```

The best validation checkpoint is written to `models/cnn_best.pth` or `models/resnet18_best.pth`. Training history is written below `results/<model>/history.json`.

## Evaluate

```bash
python evaluate.py --model cnn --data-dir data/raw/GTSRB
python evaluate.py --model resnet18 --data-dir data/raw/GTSRB
```

Evaluation writes real metrics and plots under `results/<model>/`:

- `metrics.json`
- `classification_report.json`
- `confusion_matrix.png`
- `accuracy.png`
- `loss.png`

No accuracy or visualization is generated without a completed real run.

## Run the interface

```bash
streamlit run app.py
```

The interface supports single-image upload, webcam capture, batch upload and model-performance pages.

## Test

```bash
pytest -q
```
