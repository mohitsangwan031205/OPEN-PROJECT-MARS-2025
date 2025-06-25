import os
import joblib
import numpy as np
import librosa
import pandas as pd

# Get the directory where this file lives (robust on all systems)
BASE_DIR = os.path.dirname(__file__)

# Load model components with safe path joining
calibrated_meta = joblib.load(os.path.join(BASE_DIR, "calibrated_meta.pkl"))
meta_lgb = joblib.load(os.path.join(BASE_DIR, "meta_lgb.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
thresholds = joblib.load(os.path.join(BASE_DIR, "best_thresholds.pkl"))
vote_weights = joblib.load(os.path.join(BASE_DIR, "voting_weights.pkl"))

rf_best = joblib.load(os.path.join(BASE_DIR, "rf_best.pkl"))
xgb_best = joblib.load(os.path.join(BASE_DIR, "xgb_best.pkl"))
lgb_best = joblib.load(os.path.join(BASE_DIR, "lgb_best.pkl"))

top_features = joblib.load(os.path.join(BASE_DIR, "selected_features_lgb.pkl"))

# Feature name list (must match training)
full_feature_names = (
    [f"mfcc_{i}" for i in range(40)] +
    [f"chroma_{i}" for i in range(12)] +
    [f"mel_{i}" for i in range(128)] +
    [f"contrast_{i}" for i in range(7)] +
    [f"tonnetz_{i}" for i in range(6)] +
    ["zcr", "centroid", "rmse"]
)

# Extract features from a .wav file
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))

    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz, [zcr], [centroid], [rmse]])
    return features.reshape(1, -1)

# Predict emotion from a single file
def predict_emotion(file_path):
    features = extract_features(file_path)
    df_feat = pd.DataFrame(features, columns=full_feature_names)
    selected_features = df_feat[top_features].values

    probs_rf = rf_best.predict_proba(selected_features)
    probs_xgb = xgb_best.predict_proba(selected_features)
    probs_lgb = lgb_best.predict_proba(selected_features)

    conf_max = np.max(selected_features, axis=1).reshape(-1, 1)
    conf_std = np.std(selected_features, axis=1).reshape(-1, 1)

    meta_input = np.hstack([probs_rf, probs_xgb, probs_lgb, conf_max, conf_std])

    prob1 = calibrated_meta.predict_proba(meta_input)
    prob2 = meta_lgb.predict_proba(meta_input)
    final_probs = vote_weights[0] * prob1 + vote_weights[1] * prob2

    passed = [i for i, p in enumerate(final_probs[0]) if p >= thresholds[i]]
    if passed:
        final_class = passed[np.argmax([final_probs[0][i] for i in passed])]
    else:
        final_class = np.argmax(final_probs[0])

    return le.inverse_transform([final_class])[0]

