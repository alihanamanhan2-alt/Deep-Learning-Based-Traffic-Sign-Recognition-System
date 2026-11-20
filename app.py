import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from predict import predict_image

st.set_page_config(page_title="Жол белгілерін тану жүйесі", layout="wide")
st.title("🚦 Терең оқытуға негізделген жол белгілерін тану")
st.write("Жол белгісі бар суретті жүктеңіз немесе камерадан сурет алыңыз.")

method = st.radio("Кіріс әдісі", ["Файлдан жүктеу", "Камера"])
image = None

if method == "Файлдан жүктеу":
    uploaded_file = st.file_uploader("Жол белгісі суретін таңдаңыз", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
else:
    camera_image = st.camera_input("Камерадан сурет алыңыз")
    if camera_image:
        pil_image = Image.open(camera_image).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

if image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Жүктелген сурет", use_container_width=True)

    with col2:
        try:
            result = predict_image(image)
            st.subheader("Тану нәтижесі")
            st.metric("Белгі", result["label"])
            st.metric("Сенімділік", f"{result['confidence']:.2%}")

            top_count = min(5, len(result["probabilities"]))
            top_ids = np.argsort(result["probabilities"])[::-1][:top_count]
            names = result["label"]

            st.write("Ең ықтимал сыныптар:")
            for index in top_ids:
                probability = float(result["probabilities"][index])
                st.progress(probability, text=f"Сынып {index}: {probability:.2%}")

        except Exception as error:
            st.error(str(error))
