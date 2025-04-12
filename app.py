import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Category labels
categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
img_size = 100

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/garbage_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.set_page_config(page_title="Garbage Classification", layout="centered")
st.title("🗑️ Garbage Classification using TinyML (TFLite)")

uploaded_file = st.file_uploader("Upload an image of garbage", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((img_size, img_size))
    input_data = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    st.markdown(f"### 🧠 Predicted Class: `{categories[prediction]}`")
