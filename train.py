import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 25
DATA_DIR = "data"
MODEL_DIR = "models"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    train_dir = os.path.join(DATA_DIR, "Train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            "data/Train каталогын табу мүмкін емес. GTSRB деректерін README нұсқаулығына сай орналастырыңыз."
        )

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2)
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED
    )

    validation_data = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED
    )

    if train_data.num_classes < 2:
        raise ValueError("Оқыту деректерінде кемінде екі сынып болуы керек.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "traffic_sign_model.keras")
    model = build_model(train_data.num_classes)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=callbacks
    )

    model.save(model_path)

    with open(os.path.join(MODEL_DIR, "train_history.json"), "w", encoding="utf-8") as file:
        json.dump(history.history, file, ensure_ascii=False)

    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w", encoding="utf-8") as file:
        json.dump(train_data.class_indices, file, ensure_ascii=False)

    print("Модель сақталды:", model_path)
    print("Сыныптар саны:", train_data.num_classes)


if __name__ == "__main__":
    main()
