import numpy as np
import pytest
from PIL import Image

from src.model import build_model
from src.preprocessing import build_transforms


def test_custom_cnn_output_shape():
    import torch
    model = build_model("cnn", num_classes=43)
    output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 43)


def test_resnet18_output_shape():
    import torch
    model = build_model("resnet18", num_classes=43, pretrained=False)
    output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 43)


def test_preprocessing_output_shape():
    import torch
    train_transform, eval_transform = build_transforms(64)
    image = Image.fromarray(np.zeros((100, 120, 3), dtype=np.uint8), "RGB")
    tensor = eval_transform(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 64, 64)


def test_checkpoint_prediction_guard(tmp_path):
    from predict import TrafficSignPredictor
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError):
        TrafficSignPredictor(missing)
