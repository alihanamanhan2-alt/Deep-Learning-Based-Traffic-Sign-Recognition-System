"""CNN and ResNet18 classifiers for 43 GTSRB classes."""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class TrafficSignCNN(nn.Module):
    """Compact CNN trained from scratch on 64x64 RGB images."""
    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_model(name: str, num_classes: int = 43, pretrained: bool = False):
    """Build a supported model. Pretrained ResNet weights may require internet."""
    if name.lower() == "cnn":
        return TrafficSignCNN(num_classes)
    if name.lower() == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model '{name}'. Choose 'cnn' or 'resnet18'.")
