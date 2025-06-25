import streamlit as st
from inference import predict_emotion
import tempfile
import os

st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.title("Speech Emotion Classifier")
st.markdown("Upload a `.wav` audio file and get the predicted emotion instantly!")

# File uploader
uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    # Predict emotion
    with st.spinner("Predicting..."):
        try:
            emotion = predict_emotion(tmp_path)
            st.success(f" Predicted Emotion: **{emotion.upper()}**")
        except Exception as e:
            st.error(f"Error: {e}")

    # Clean up
    os.remove(tmp_path)
