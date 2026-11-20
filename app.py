import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import MODEL_DIR, RESULTS_DIR
from predict import TrafficSignPredictor

HISTORY_PATH = RESULTS_DIR / "recognition_history.json"

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="wide")
st.title("🚦 Deep-Learning-Based Traffic Sign Recognition System")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Recognition", "Webcam", "Batch Recognition", "Model Performance", "About Project"],
)

checkpoint_files = sorted(MODEL_DIR.glob("*_best.pth"))


def available_checkpoint():
    if not checkpoint_files:
        return None
    selected = st.sidebar.selectbox(
        "Model", [p.stem.removesuffix("_best") for p in checkpoint_files]
    )
    return MODEL_DIR / f"{selected}_best.pth"


def add_history(entries):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if HISTORY_PATH.is_file():
        previous = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    previous.extend(entries)
    HISTORY_PATH.write_text(json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")


if page == "Home":
    st.header("GTSRB traffic sign classification")
    st.write("CNN and ResNet18 system implemented with PyTorch, OpenCV, scikit-learn and Streamlit.")
    if checkpoint_files:
        st.success(f"Found {len(checkpoint_files)} trained checkpoint(s).")
    else:
        st.warning("No trained .pth weights found. Run `python train.py --model cnn` first.")

elif page in {"Recognition", "Webcam", "Batch Recognition"}:
    weights = available_checkpoint()
    if weights is None:
        st.error("No trained model is available. Train a model before recognition.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def load_predictor(path):
        return TrafficSignPredictor(Path(path), device="auto")

    predictor = load_predictor(str(weights))

    if page == "Recognition":
        uploaded = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png", "ppm", "bmp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded image", use_container_width=True)
            result = predictor.predict_image(image)
            with col2:
                st.subheader("Recognition result")
                st.metric("Predicted sign", result["label"])
                st.metric("Confidence", f"{result['confidence']:.2%}")
                st.write("Top predictions")
                st.dataframe(pd.DataFrame(result["top_predictions"]), use_container_width=True)
            add_history([{
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": predictor.model_name,
                "class_id": result["class_id"],
                "label": result["label"],
                "confidence": result["confidence"],
            }])

    elif page == "Webcam":
        camera_image = st.camera_input("Capture a traffic sign")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            result = predictor.predict_image(image)
            st.image(image, use_container_width=True)
            st.success(f"{result['label']} — {result['confidence']:.2%}")

    else:
        uploads = st.file_uploader(
            "Upload multiple images", type=["jpg", "jpeg", "png", "ppm", "bmp"], accept_multiple_files=True
        )
        if uploads:
            rows = []
            history_entries = []
            cols = st.columns(3)
            for index, uploaded in enumerate(uploads):
                image = Image.open(uploaded).convert("RGB")
                result = predictor.predict_image(image)
                with cols[index % 3]:
                    st.image(image, caption=uploaded.name, use_container_width=True)
                    st.write(f"**{result['label']}**")
                    st.write(f"Confidence: {result['confidence']:.2%}")
                rows.append({"Filename": uploaded.name, **result})
                history_entries.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model": predictor.model_name,
                    "source": uploaded.name,
                    "class_id": result["class_id"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            add_history(history_entries)

elif page == "Model Performance":
    st.header("Model evaluation results")
    for model_dir in sorted((RESULTS_DIR).glob("*")):
        if not model_dir.is_dir():
            continue
        metrics_path = model_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.subheader(metrics["model"])
            st.json(metrics)
            confusion = model_dir / "confusion_matrix.png"
            accuracy = model_dir / "accuracy.png"
            loss = model_dir / "loss.png"
            if confusion.is_file():
                st.image(str(confusion), caption="Confusion matrix")
            if accuracy.is_file() and loss.is_file():
                c1, c2 = st.columns(2)
                c1.image(str(accuracy), caption="Accuracy")
                c2.image(str(loss), caption="Loss")
    if not any((p / "metrics.json").is_file() for p in RESULTS_DIR.glob("*")):
        st.info("No evaluated metrics yet. Run `python evaluate.py --model cnn` and/or `--model resnet18`.")

else:
    st.header("About Project")
    st.write("**Dataset:** GTSRB, 43 traffic sign classes")
    st.write("**Models:** custom CNN and pretrained or from-scratch ResNet18")
    st.write("**Framework:** PyTorch")
    st.write("**Interface:** Streamlit")
