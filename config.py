"""Project configuration defaults for training and inference."""
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "GTSRB"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
NUM_CLASSES = 43
IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42
