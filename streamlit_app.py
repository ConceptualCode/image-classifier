import os
import requests
import streamlit as st
from PIL import Image
import io
import time

# Read the API URL from the environment variable
api_url = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Chess Piece Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if image.mode == "RGBA":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        files = {"file": ("image.jpg", buffered.getvalue(), "image/jpeg")}
        
        start_time = time.time()
        response = requests.post(api_url, files=files)
        latency = time.time() - start_time
        throughput = 1 / latency if latency > 0 else float('inf')

        if response.status_code == 200:
            prediction = response.json()
            st.write("Prediction:", prediction["class_label"])
            st.write("Confidence:", f"{prediction['confidence']:.2f}")
            st.write("Latency:", f"{latency:.2f} seconds")
            st.write("Throughput:", f"{throughput:.2f} predictions per second")
        else:
            st.error(f"Error in classification request.\n\nStatus Code: {response.status_code}\nResponse Text: {response.text}")