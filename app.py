from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Apple Leaf Disease Detector", page_icon="🍏")

BASE = Path(__file__).parent
LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
ICONS = {"healthy": "✅", "multiple_diseases": "⚠️", "rust": "🟠", "scab": "🟤"}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE / "plant_model.keras", custom_objects={"preprocess_input": preprocess_input})


model = load_model()

st.title("🍏 Apple Leaf Disease Detector")
st.write(
    "A ResNet50 transfer-learning model (trained on the Kaggle Plant Pathology 2020 dataset) classifies a photo "
    "of an apple tree leaf as healthy, showing multiple diseases, rust, or scab."
)

source = st.radio("Image source", ["Upload a photo", "Use my camera"], horizontal=True)
file = st.file_uploader("Upload a leaf photo", type=["png", "jpg", "jpeg"]) if source == "Upload a photo" else st.camera_input("Take a photo")

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf", width=250)

    resized = img.resize((224, 224))
    x = np.array(resized, dtype=np.float32)[np.newaxis, ...]

    if st.button("Diagnose"):
        proba = model.predict(x, verbose=0)[0]
        pred = LABELS[int(np.argmax(proba))]
        st.success(f"{ICONS[pred]} Predicted: **{pred.replace('_', ' ')}**  ({proba.max():.1%} confidence)")
        st.bar_chart({"probability": dict(zip(LABELS, proba))}["probability"])

st.caption(
    "Model: ResNet50 (frozen, pretrained on ImageNet) + a small trained classification head (validation mean AUC "
    "≈ 0.98 on the Kaggle data). 'multiple_diseases' is the rarest category in the training data (91 out of 1,821 "
    "images), so it is the hardest one for the model to recognize confidently."
)
