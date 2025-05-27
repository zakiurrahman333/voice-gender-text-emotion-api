import librosa
import numpy as np
import joblib

import os

model_path = os.path.join(os.path.dirname(__file__), "..", "gmm_gender_model.pkl")
model_path = os.path.abspath(model_path)
model = joblib.load(model_path)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.hstack([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0)
    ])
    return combined

def predict_gender(audio_path):
    feat = extract_features(audio_path).reshape(1, -1)

    male_score = model['male'].score(feat)
    female_score = model['female'].score(feat)

    gender = 'male' if male_score > female_score else 'female'
    return gender
