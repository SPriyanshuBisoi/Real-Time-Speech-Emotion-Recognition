# feature_extraction.py
import numpy as np
import librosa

def extract_features(audio_path):
    X, sample_rate = librosa.load(audio_path, sr=22050, duration=2.5, offset=0.5)
    result = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))
    return result.reshape(1, -1)
