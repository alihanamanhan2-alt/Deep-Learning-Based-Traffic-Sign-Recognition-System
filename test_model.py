import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join("models", "traffic_sign_model.keras")
TEST_DIR = os.path.join("data", "Test")


def main():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError("data/Test каталогын табу мүмкін емес.")

    model = tf.keras.models.load_model(MODEL_PATH)

    test_images = []
    test_labels = []

    import pandas as pd
    from PIL import Image

    csv_path = os.path.join("data", "Test.csv")
    if os.path.exists(csv_path):
        table = pd.read_csv(csv_path)
        for _, row in table.iterrows():
            path = os.path.join("data", str(row["Path"]))
            if not os.path.exists(path):
                continue
            image = np.array(Image.open(path).convert("RGB").resize((32, 32)))
            test_images.append(image)
            test_labels.append(int(row["ClassId"]))
    else:
        raise FileNotFoundError("data/Test.csv файлы табылмады.")

    x_test = np.array(test_images, dtype="float32") / 255.0
    y_test = np.array(test_labels)

    loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, model.output_shape[-1]), verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print("Тестілік дәлдік:", round(accuracy * 100, 2), "%")
    print("Жоғалтулар:", round(loss, 4))
    print("Матрица:")
    print(confusion_matrix(y_test, predictions))
    print("Классификация есебі:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
