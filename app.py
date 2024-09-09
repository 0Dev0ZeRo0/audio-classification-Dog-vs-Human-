
import streamlit as st
import librosa
import numpy as np
import joblib

# Load the saved model
model = joblib.load('best_model.pkl')

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

st.title("Human vs Dog Voice Classification")
audio_file = st.file_uploader("Upload an audio file[.wav format only]", type=["wav"])

if audio_file is not None:
    # Save the uploaded file to a temporary directory
    with open("temp.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # Extract features
    features = extract_features("temp.wav").reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    st.write(f"Prediction: {prediction[0]}")
