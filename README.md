# Emotion Classification from Speech Audio
**MARS Open Projects 2025 – AI/ML Project 1 Submission**

This project focuses on building an end-to-end traditional machine learning pipeline to classify emotions from raw speech `.wav` audio files. The final solution is a confidence-aware, threshold-optimized, soft-voting stacked model trained on engineered acoustic features, achieving excellent real-world performance.

## Project Overview

Humans express emotions not just through words, but through how they speak — pitch, tone, pace, and frequency content. This project leverages that information to classify six emotions using `.wav` audio data:

> `happy`, `sad`, `angry`, `calm`, `fearful`, `neutral`

We built a high-impact traditional ML solution using handcrafted features, calibrated probabilities, ensemble learning, and robust evaluation.

## Dataset

- **Source:** `Audio_Speech_Actors_1-24` and `Audio_Song_Actors_1-24`
- **Files:** Raw `.wav` audio files
- **Split:** 80/20 train-validation
- **Classes used:** `happy`, `sad`, `angry`, `calm`, `fearful`, `neutral`
- **Classes dropped:** `disgust`, `surprised` (due to data sparsity and imbalance)

## Feature Engineering

Using **Librosa**, we extracted the following features for each audio file:

- **MFCCs** (40 coefficients)
- **Chroma STFT**
- **Mel Spectrogram**
- **Spectral Contrast**
- **Tonnetz**
- **Zero Crossing Rate**
- **Spectral Centroid**

Each audio file was converted into a **56-dimensional feature vector**. Additional **meta features** were later added during stacking.

## Exploratory Data Analysis (EDA)

- Plotted emotion distribution to assess class imbalance
- Analyzed feature distributions per emotion
- Visualized correlation heatmaps to identify redundant or impactful features
- Found that tonal features like MFCCs and spectral contrast vary significantly across emotion classes

## Modeling Pipeline

The modeling process involved multiple stages:

### Base Model Training
- Trained three base models:  
  **Random Forest**, **XGBoost**, and **LightGBM**  
- Hyperparameter tuning performed using `RandomizedSearchCV` with **F2-score** as the scoring metric
- SMOTE applied to handle class imbalance in training data

### Meta Feature Generation
- From base models’ predicted probabilities, created meta features:
  - All class probabilities from each model
  - **Confidence max** (highest predicted prob)
  - **Confidence std** (spread of predicted probs)

### Stacked Ensemble Modeling
- Meta learners used:
  - **Calibrated Logistic Regression**
  - **LightGBM**
- Final prediction via **soft voting** between the two meta models (weights: 0.6, 0.4)

### Threshold Tuning
- Per-class thresholds were optimized using `fbeta_score(beta=2)` on validation predictions to better control recall vs precision.

### Evaluation
- Predicted on validation set using stacked model and tuned thresholds
- Calculated:
  - Accuracy
  - F1 score (weighted)
  - F2 score (weighted)
  - Per-class accuracy
  - Confusion matrix

## Final Evaluation Metrics

| Metric                  | Value         |
|-------------------------|---------------|
| Overall Accuracy        | **79.18%**     |
| Weighted F1 Score       | **79.23%**     |
| Weighted F2 Score       | **79.18%**     |
| Fearful Class Accuracy  | **76.00%**     |
| Happy Class Accuracy    | **77.33%**     |
| Sad Class Accuracy      | **75.00%**     |
| Angry Class Accuracy    | **85.33%**     |
| Neutral Class Accuracy  | **86.84%**     |
| Calm Class Accuracy     | **78.67%**     |

## Final Model Details

- **Name:** `Soft Voting Meta Ensemble (Calibrated LR + LGBM)`
- **Input Features:**  
  - Predicted class probabilities from base models  
  - Confidence max & std
- **Balancing Strategy:** SMOTE
- **Threshold Tuning:** Optimized for each class
- **Model Calibration:** CalibratedClassifierCV for probabilistic output

## Project Structure

EmotionClassifier/
├── data/ # Original audio dataset (optional)
├── notebooks and models/
│ ├── MARS Project.ipynb # Main notebook for training & evaluation
│ ├── rf_best.pkl # Base model 1
│ ├── xgb_best.pkl # Base model 2
│ ├── lgb_best.pkl # Base model 3
│ ├── final_model_softvote/ # Folder for final meta-models
│ │ ├── calibrated_meta.pkl
│ │ ├── meta_lgb.pkl
│ │ ├── best_thresholds.pkl
│ │ ├── voting_weights.pkl
│ │ └── label_encoder.pkl
│ └── selected_features_lgb.pkl
├── scripts/
│ ├── inference.py # Predict on a single audio file
│ └── test_model.py # Predict on a folder of audio files
├── test_wavs/ # Folder to place test .wav files
├── predictions.csv # Output predictions file
└── README.md # Project documentation



## Testing on Custom Audio Files

###  Option 1: Predict a Folder of `.wav` Files

Place your test `.wav` files inside `test_wavs/`, then run:

```python
from test_model import test_model_on_folder
test_model_on_folder("test_wavs/")
# output will be saved to predictions.csv

### option 2 : predict single emotion

```python
from inference import predict_emotion
emotion = predict_emotion("test_wavs/sample.wav")
print("Predicted Emotion:", emotion)

### For Running the app

```python
# From EmotionClassifier/scripts/
streamlit run app.py
