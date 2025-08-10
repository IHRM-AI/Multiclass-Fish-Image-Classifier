import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("MobileNet_final.h5")
    return model

model = load_model()

# App title
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish to predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence_scores = predictions * 100

    # Display results
    st.subheader(f"Predicted Class: **{predicted_class}**")
    st.write("### Confidence Scores:")
    for cls, score in zip(class_names, confidence_scores):
        st.write(f"{cls}: {score:.2f}%")
