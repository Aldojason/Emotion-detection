"""
utils.py
Shared utility functions for the Facial Emotion Recognition project.
Includes dataset detection/loading helpers, model architecture builder,
and common constants used across training, prediction, and evaluation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout,
    Flatten, Dense, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 20

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

EMOTION_LABELS_DISPLAY = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
}

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")
DATASET_CSV_PATH = os.path.join(DATASET_DIR, "fer2013.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Dataset detection
# ---------------------------------------------------------------------------
def detect_dataset_format():
    """
    Detects whether the dataset is available as a folder structure
    (dataset/train/<emotion>/*.jpg) or as a single fer2013.csv file.

    Returns:
        str: "folder", "csv", or "none"
    """
    if os.path.isdir(DATASET_TRAIN_DIR):
        has_images = False
        for emotion in EMOTION_LABELS:
            emotion_path = os.path.join(DATASET_TRAIN_DIR, emotion)
            if os.path.isdir(emotion_path) and len(os.listdir(emotion_path)) > 0:
                has_images = True
                break
        if has_images:
            return "folder"

    if os.path.isfile(DATASET_CSV_PATH):
        return "csv"

    return "none"


def load_csv_dataset():
    """
    Loads the fer2013.csv dataset into numpy arrays for training/testing.

    Returns:
        (X_train, y_train, X_test, y_test): numpy arrays
    """
    if not os.path.isfile(DATASET_CSV_PATH):
        raise FileNotFoundError(
            f"Dataset CSV not found at {DATASET_CSV_PATH}. "
            "Please place fer2013.csv inside the dataset folder."
        )

    df = pd.read_csv(DATASET_CSV_PATH)

    required_cols = {"emotion", "pixels"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            "fer2013.csv must contain 'emotion' and 'pixels' columns."
        )

    pixels = df["pixels"].tolist()
    images = []
    for pixel_seq in pixels:
        try:
            face = np.array(pixel_seq.split(" "), dtype="float32")
            face = face.reshape(IMG_SIZE, IMG_SIZE)
            images.append(face)
        except ValueError:
            # Skip corrupted rows
            continue

    images = np.array(images, dtype="float32")
    images = np.expand_dims(images, -1) / 255.0

    labels = to_categorical(df["emotion"].values[: len(images)], NUM_CLASSES)

    if "Usage" in df.columns:
        usage = df["Usage"].values[: len(images)]
        train_mask = usage == "Training"
        test_mask = ~train_mask
        X_train, y_train = images[train_mask], labels[train_mask]
        X_test, y_test = images[test_mask], labels[test_mask]
    else:
        split_idx = int(len(images) * 0.8)
        X_train, y_train = images[:split_idx], labels[:split_idx]
        X_test, y_test = images[split_idx:], labels[split_idx:]

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
def build_emotion_model():
    """
    Builds and compiles the CNN architecture used for emotion recognition.

    Returns:
        tensorflow.keras.Model: compiled Keras model
    """
    model = Sequential(name="EmotionCNN")

    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    # Block 1
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Face preprocessing helpers
# ---------------------------------------------------------------------------
def preprocess_face(face_img):
    """
    Prepares a cropped face image (BGR or grayscale) for model prediction.

    Args:
        face_img (np.ndarray): cropped face region

    Returns:
        np.ndarray: preprocessed image of shape (1, 48, 48, 1)
    """
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


def load_face_cascade():
    """
    Loads the OpenCV Haar Cascade face detector.

    Returns:
        cv2.CascadeClassifier
    """
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if cascade.empty():
        raise IOError(
            "Failed to load Haar Cascade classifier. "
            "Check your OpenCV installation."
        )
    return cascade


def ensure_dirs():
    """Ensures required project directories exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
