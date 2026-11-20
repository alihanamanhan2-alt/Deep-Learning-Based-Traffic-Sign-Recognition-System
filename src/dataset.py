"""GTSRB datasets and deterministic data-loader construction."""
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE, SEED
from src.preprocessing import build_transforms

GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 metric tons",
]


class GTSRBTestDataset(Dataset):
    """Read official GTSRB Test.csv (semicolon-separated) and image files."""
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        csv_path = self.root / "Test.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"GTSRB test annotation file not found: {csv_path}")
        table = pd.read_csv(csv_path, sep=None, engine="python")
        if not {"Path", "ClassId"}.issubset(table.columns):
            raise ValueError("Test.csv must contain Path and ClassId columns")
        self.samples = []
        for row in table.itertuples(index=False):
            image_path = self.root / str(getattr(row, "Path"))
            label = int(getattr(row, "ClassId"))
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in Test.csv not found: {image_path}")
            if not 0 <= label < len(GTSRB_CLASS_NAMES):
                raise ValueError(f"Invalid GTSRB class id {label} in {csv_path}")
            self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_dir: Path = DATA_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = 0.2,
    seed: int = SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """Create stratified train/validation loaders and official test loader.

    Expected layout: data_dir/Train/<class_id>/*.ppm and data_dir/Test.csv.
    """
    data_dir = Path(data_dir)
    train_root = data_dir / "Train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_root}")
    train_transform, eval_transform = build_transforms(image_size)
    augmented = ImageFolder(train_root, transform=train_transform)
    plain = ImageFolder(train_root, transform=eval_transform)
    if len(augmented.classes) != len(GTSRB_CLASS_NAMES):
        raise ValueError(
            f"Expected 43 class folders in {train_root}, found {len(augmented.classes)}"
        )
    if augmented.classes != [str(i) for i in range(43)]:
        raise ValueError("GTSRB class folders must be named 0 through 42")
    labels = [label for _, label in augmented.samples]
    indices = list(range(len(labels)))
    train_indices, val_indices = train_test_split(
        indices, test_size=validation_fraction, random_state=seed,
        stratify=labels,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(augmented, train_indices), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator,
    )
    val_loader = DataLoader(
        Subset(plain, val_indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        GTSRBTestDataset(data_dir, eval_transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    class_names = {i: name for i, name in enumerate(GTSRB_CLASS_NAMES)}
    return train_loader, val_loader, test_loader, class_names
