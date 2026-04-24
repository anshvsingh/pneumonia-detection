import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="anshvsingh/pneumonia-detection",   
        filename="xray_model.hdf5"
    )
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner("Model is being loaded from Hugging Face..."):
    model = load_model()


class_names = ["Normal", "Pneumonia"]  


st.write("# Pneumonia Identification System")
st.write("Upload a chest X-ray image and the model will predict whether it shows signs of Pneumonia.")

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)   
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)          
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.info("Please upload a chest X-ray image to get started.")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)                

    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    if predicted_class == "Pneumonia":
        st.error(
            f"⚠️ **Pneumonia Detected** — The model is **{confidence:.2f}%** confident.\n\n"
            "Please consult a medical professional for proper diagnosis."
        )
    else:
        st.success(
            f"✅ **Normal** — The model is **{confidence:.2f}%** confident.\n\n"
            "No signs of pneumonia detected."
        )
