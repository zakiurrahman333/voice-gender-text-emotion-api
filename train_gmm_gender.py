import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import joblib

# Folder structure:
# data/
#   male/
#   female/

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

def load_data(data_dir):
    features = []
    labels = []
    for label, gender in enumerate(['male', 'female']):
        gender_dir = os.path.join(data_dir, gender)
        for file_name in os.listdir(gender_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(gender_dir, file_name)
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(label)
    return np.array(features), np.array(labels)  # ✅ Fix: return both

def train_and_save_model(data_dir, model_path='gmm_gender_model.pkl'):
    X, y = load_data(data_dir)
    male_features = X[y == 0]
    female_features = X[y == 1]

    gmm_male = GaussianMixture(n_components=8, covariance_type='diag', random_state=42)
    gmm_female = GaussianMixture(n_components=8, covariance_type='diag', random_state=42)


    gmm_male.fit(male_features)
    gmm_female.fit(female_features)

    # Save models in a dict
    model = {'male': gmm_male, 'female': gmm_female}
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_dir = 'dataset'
    train_and_save_model(data_dir)
