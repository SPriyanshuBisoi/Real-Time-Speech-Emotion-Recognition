import streamlit as st
import numpy as np
import tensorflow as tf
import soundfile as sf
import tempfile
import os
import sounddevice as sd
from feature_extraction import extract_features

# Load the trained model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'calm']

# Set page configuration
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤")

# Add custom CSS for font sizes and divider
# Custom CSS
st.markdown("""
    <style>
        .title {
            font-size: 33px !important;
            font-weight: 600;
            margin-bottom: 50px;
        }
        .section-header {
            font-size: 24px !important;
            margin-bottom: 10px;
        }
        .stButton>button {
            font-size: 16px !important;
        }
        .stSpinner {
            font-size: 18px !important;
        }
        .emotion-result {
            font-size: 22px !important;
            font-weight: bold;
            color: #4CAF50;
        }
        .divider-col {
            border-left: 2px solid #ccc;
            height: 100%;
            padding-left: 20px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🎤 Real-Time Emotion Recognition from Speech</div>', unsafe_allow_html=True)

col1, divider, col2 = st.columns([1, 0.5, 1])

with col1:
    # Upload Section
    st.markdown('<div class="section-header">📁 Upload Audio File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_file is not None:
        if st.button("📊 Predict from Upload", key="upload_btn"):
            with st.spinner("Processing uploaded audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    tmpfile.write(uploaded_file.read())
                    tmpfile_path = tmpfile.name
                features = extract_features(tmpfile_path)
                prediction = model.predict(features)
                predicted_emotion = emotions[np.argmax(prediction)]
                os.remove(tmpfile_path)
            st.markdown(f"<div class='emotion-result'>🎭 Predicted Emotion: {predicted_emotion.capitalize()}</div>", unsafe_allow_html=True)

with divider:
    st.markdown('<div class="divider-col"></div>', unsafe_allow_html=True)

# -------- Right Column: Record Section --------
with col2:
    st.markdown('<div class="section-header">🎙️ Record Audio</div>', unsafe_allow_html=True)

    fs = 22050
    duration = 3

    if "recording" not in st.session_state:
        st.session_state.recording = False
        st.session_state.audio_data = None
        st.session_state.recorded_file = None
        st.session_state.predicted_emotion = None

    if st.button("🔴 Start Recording"):
        st.session_state.audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            st.session_state.recorded_file = tmpfile.name
            sf.write(tmpfile.name, st.session_state.audio_data, fs)
        st.success("✅ Recording completed. Ready to process.")

    if st.session_state.recorded_file:
        if st.button("🎧 Process & Predict"):
            with st.spinner("Processing recorded audio..."):
                features = extract_features(st.session_state.recorded_file)
                prediction = model.predict(features)
                st.session_state.predicted_emotion = emotions[np.argmax(prediction)]
                try:
                    os.remove(st.session_state.recorded_file)
                except Exception as e:
                    st.warning(f"Couldn't delete temp file: {e}")
                st.session_state.recorded_file = None

    if st.session_state.predicted_emotion:
        st.markdown(f"<div class='emotion-result'>🎭 Predicted Emotion: {st.session_state.predicted_emotion.capitalize()}</div>", unsafe_allow_html=True)
        if st.button("🔁 Clear"):
            st.session_state.clear()
            st.success("✅ Cleared! Ready for new recording.")
