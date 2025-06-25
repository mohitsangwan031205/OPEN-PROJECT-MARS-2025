
# 🎤 Emotion Classification from Speech Audio

**MARS Open Projects 2025 – AI/ML Project 1 Submission**

This project focuses on building an end-to-end traditional machine learning pipeline to classify emotions from raw speech `.wav` audio files. The final solution is a confidence-aware, threshold-optimized, soft-voting stacked model trained on engineered acoustic features, achieving excellent real-world performance.

🌐 **Try the Web App:**  
(https://open-project-mars-2025-9vksaschnnvvfg9uepwami.streamlit.app/) 


---

## 📌 Project Overview

Humans express emotions not just through words, but also through *how* they speak — pitch, tone, pace, and frequency content. This project leverages acoustic features from `.wav` files to classify six core emotions:

> `happy`, `sad`, `angry`, `calm`, `fearful`, `neutral`

---

## 🎧 Dataset

- **Source:** `Audio_Speech_Actors_1-24` and `Audio_Song_Actors_1-24`
- **Format:** `.wav` files
- **Split:** 80% training / 20% validation
- **Used Classes:** `happy`, `sad`, `angry`, `calm`, `fearful`, `neutral`
- **Dropped:** `disgust`, `surprised` (due to sparsity)

---

## 🧪 Feature Engineering

Using `librosa`, the following features were extracted from each `.wav` file:

- MFCCs (40 coefficients)
- Chroma STFT (12)
- Mel Spectrogram (128)
- Spectral Contrast (7)
- Tonnetz (6)
- Zero Crossing Rate (1)
- Spectral Centroid (1)
- Root Mean Square Energy (1)

Each sample was represented as a **56-dimensional feature vector**. Meta features like **confidence max** and **confidence std** were added during stacking.

---

## 📊 EDA Highlights

- Emotion distribution plots revealed class imbalance
- Feature distributions varied by emotion
- Heatmaps helped identify redundant features
- MFCC and spectral contrast were highly discriminative

---

## 🧠 Modeling Pipeline

### 1️⃣ Base Models

- Random Forest  
- XGBoost  
- LightGBM  

All base models were tuned with `RandomizedSearchCV` using **F2-score** as the metric. **SMOTE** was used for oversampling.

### 2️⃣ Meta Features

From the base models, the following were computed:

- All class probabilities (for each model)
- `confidence_max`: highest prob from any base model
- `confidence_std`: standard deviation of probs

### 3️⃣ Meta Models (Stacked Ensemble)

- Calibrated Logistic Regression (via `CalibratedClassifierCV`)
- LightGBM

Final prediction via **soft voting**:
- `60%` weight to Calibrated LR
- `40%` weight to LightGBM meta model

### 4️⃣ Threshold Tuning

Per-class probability thresholds were optimized using `fbeta_score(beta=2)` for better recall on harder classes.

---

## 📈 Final Evaluation Metrics

| Metric                  | Value        |
|-------------------------|--------------|
| **Overall Accuracy**    | 79.18%       |
| Weighted F1 Score       | 79.23%       |
| Weighted F2 Score       | 79.18%       |
| Angry Accuracy          | 85.33%       |
| Neutral Accuracy        | 86.84%       |
| Calm Accuracy           | 78.67%       |
| Happy Accuracy          | 77.33%       |
| Sad Accuracy            | 75.00%       |
| Fearful Accuracy        | 76.00%       |

---

## 🗂️ Project Structure

```text
open-project-mars-2025/
├── scripts and models/
│   ├── app.py                    # Streamlit app
│   ├── inference.py              # Single file prediction
│   ├── test_model.py             # Batch predictions
│   ├── calibrated_meta.pkl       # Meta model 1
│   ├── meta_lgb.pkl              # Meta model 2
│   ├── best_thresholds.pkl       # Class-wise thresholds
│   ├── voting_weights.pkl        # Soft voting weights
│   ├── label_encoder.pkl         # Label encoder
│   ├── rf_best.pkl               # Base model 1
│   ├── xgb_best.pkl              # Base model 2
│   ├── lgb_best.pkl              # Base model 3
│   ├── selected_features_lgb.pkl # Selected features for meta model
│   └── predictions.csv           # Output predictions (batch)
├── test_wavs/                    # Folder to place test .wav files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
````

---

## 🧪 Use the Model

You can use the trained model in two ways:

### ▶️ Predict a Single Audio File

```python
from inference import predict_emotion

emotion = predict_emotion("test_wavs/sample.wav")
print("Predicted Emotion:", emotion)
```

### 📁 Predict Emotions for a Folder of Files

```python
from test_model import test_model_on_folder

test_model_on_folder("test_wavs/")
# Output saved to predictions.csv
```

---

## 🚀 Run the Web App Locally

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Start the App

```bash
# Inside scripts and models/
streamlit run app.py
```

---



## 📃 Example `requirements.txt`

```txt
streamlit
numpy
pandas
librosa
scikit-learn
xgboost
lightgbm
joblib
```

Make sure to include any additional libraries your project uses.

---

## 🙋‍♀️ Authors

* Mohit Kumar ( 23112061 )
* MARS Open Projects 2025
* Emotion Classification — AI/ML Track





