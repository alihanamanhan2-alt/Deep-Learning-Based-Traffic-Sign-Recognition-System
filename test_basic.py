import pytest


def test_predict_module_exists():
    from predict import get_class_names
    assert callable(get_class_names)


def test_prepare_image_shape():
    import cv2
    import numpy as np
    from predict import prepare_image

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = prepare_image(image)

    assert result.shape == (1, 32, 32, 3)
