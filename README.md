# Facial Emotion Recognition

A complete, production-quality real-time Facial Emotion Recognition system built with **TensorFlow/Keras**, **OpenCV**, and **Flask**. A convolutional neural network is trained from scratch on the **FER2013** dataset to classify seven emotions — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise — from a live webcam feed, with a dark-themed web dashboard for real-time monitoring.

---

## 1. Project Overview

- **Model**: Custom CNN (3 convolutional blocks + dense classifier), trained on 48×48 grayscale face images.
- **Face detection**: OpenCV Haar Cascade, run on each webcam frame.
- **Interfaces**:
  - `training/predict.py` — standalone OpenCV window (desktop demo, `Q` to quit, `S` to screenshot).
  - `app.py` — Flask web dashboard with live MJPEG video stream and a JSON status API.
- **Outputs**: trained model (`.h5`), training curves, confusion matrix, and a classification report.

```
EmotionRecognition/
├── dataset/                  # place FER2013 data here
│   ├── train/<emotion>/*.jpg
│   └── test/<emotion>/*.jpg
│   (or dataset/fer2013.csv)
├── models/
│   └── emotion_model.h5      # created by train.py
├── training/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── app.py
├── requirements.txt
└── README.md
```

---

## 2. Installation

Requires **Python 3.10 or 3.11** (TensorFlow 2.15 compatibility).

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** TensorFlow runs on CPU out of the box. A GPU is not required, but training will be faster with one (install `tensorflow[and-cuda]` and matching CUDA/cuDNN drivers if desired).

---

## 3. Dataset Setup

Download the **FER2013** dataset (e.g. from Kaggle: *"Facial Expression Recognition Challenge"*) and place it in the `dataset/` folder using **either** of these layouts — the code auto-detects which one is present:

**Option A — folder structure**
```
dataset/train/angry/*.jpg
dataset/train/disgust/*.jpg
dataset/train/fear/*.jpg
dataset/train/happy/*.jpg
dataset/train/neutral/*.jpg
dataset/train/sad/*.jpg
dataset/train/surprise/*.jpg
dataset/test/<same seven folders>
```

**Option B — single CSV**
```
dataset/fer2013.csv     # columns: emotion, pixels, (optional) Usage
```

---

## 4. Training

```bash
python training/train.py
```

This will:
1. Auto-detect the dataset format.
2. Build the CNN and train for up to 20 epochs with `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`.
3. Save the best model to `models/emotion_model.h5`.
4. Save `models/training_history.png` and `models/accuracy_loss.png`.

Training on CPU typically takes 30–90 minutes depending on hardware and dataset size.

---

## 5. Testing / Evaluation

```bash
python training/evaluate.py
```

Generates:
- `models/confusion_matrix.png`
- `models/classification_report.txt` (accuracy, precision, recall, F1-score per class)
- Console output with overall accuracy

---

## 6. Running the Flask Web App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. The dashboard shows:
- Live annotated webcam stream (bounding box + emotion + confidence)
- Current emotion and confidence bar
- System status (model loaded, camera connected, faces in frame, FPS)

---

## 7. Running the Standalone Webcam Demo

```bash
python training/predict.py
```

Controls:
- **Q** — quit
- **S** — save a screenshot to `screenshots/`

---

## 8. Screenshots

Screenshots taken via the standalone demo (`S` key) or Flask dashboard are saved to the `screenshots/` folder. Add your own captures here after running the app, e.g.:

```
screenshots/screenshot_20260101_120000.png
```

---

## 9. Error Handling

The system handles and reports the following conditions with clear messages instead of crashing:

| Condition | Behavior |
|---|---|
| Camera unavailable | Dashboard shows "Camera unavailable"; standalone script exits with a clear error |
| Dataset missing | `train.py` exits with setup instructions |
| Model missing | `app.py` and `predict.py` report the model is not trained yet |
| Corrupted image / CSV row | Skipped during dataset loading, training continues |
| No face detected | Overlay text "No face detected" shown on frame |
| Multiple faces | Each face is detected and annotated independently |
| Model loading errors | Caught and surfaced via the `/status` endpoint and console logs |

---

## 10. Future Improvements

- Swap Haar Cascade for a DNN-based face detector (e.g. RetinaFace, MTCNN) for better accuracy in varied lighting/poses.
- Add temporal smoothing across frames to reduce prediction flicker.
- Support multi-face emotion analytics/logging over time.
- Add a transfer-learning variant (e.g. MobileNetV2 backbone) for higher accuracy.
- Dockerize the application for one-command deployment.
- Add authentication and multi-user session support to the Flask app.

---

## License

This project is provided as-is for educational and research purposes.
