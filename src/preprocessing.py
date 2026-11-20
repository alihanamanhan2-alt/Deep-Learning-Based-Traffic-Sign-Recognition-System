"""Image transforms shared by training, evaluation and prediction."""
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 64):
    """Return augmentation and evaluation transforms.

    Horizontal flipping is intentionally omitted: flipping can change the
    meaning of directional traffic signs.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform
