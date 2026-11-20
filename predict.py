import os
import json
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (32, 32)
MODEL_PATH = os.path.join("models", "traffic_sign_model.keras")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Модель табылмады. Алдымен python train.py командасын орындаңыз."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def get_class_names():
    path = os.path.join("models", "class_indices.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        class_indices = json.load(file)

    names = [None] * len(class_indices)
    for name, index in class_indices.items():
        names[index] = name

    custom_path = os.path.join("data", "class_names.json")
    if os.path.exists(custom_path):
        with open(custom_path, "r", encoding="utf-8") as file:
            custom_names = json.load(file)
        return custom_names

    return names


def prepare_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(image):
    model = load_model()
    processed = prepare_image(image)
    probabilities = model.predict(processed, verbose=0)[0]

    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])

    names = get_class_names()
    if names is not None and class_id < len(names):
        label = names[class_id]
    else:
        label = f"Сынып {class_id}"

    return {
        "class_id": class_id,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities
    }
