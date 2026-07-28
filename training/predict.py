import os
import sys
import time
import datetime

import cv2
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    EMOTION_LABELS, EMOTION_LABELS_DISPLAY, MODEL_PATH,
    preprocess_face, load_face_cascade
)

SCREENSHOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots"
)


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please run train.py first."
            )
        print("Loading emotion recognition model...")
        self.model = load_model(model_path)

        try:
            self.face_cascade = load_face_cascade()
        except IOError as e:
            raise IOError(f"Could not load face detector: {e}")

        print("Model and face detector loaded successfully.")

    def detect_faces(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        return faces

    def predict_emotion(self, face_img):
        processed = preprocess_face(face_img)
        predictions = self.model.predict(processed, verbose=0)[0]
        idx = int(np.argmax(predictions))
        label = EMOTION_LABELS[idx]
        confidence = float(predictions[idx]) * 100.0
        return label, confidence


def run_webcam_demo(camera_index=0):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    try:
        predictor = EmotionPredictor()
    except (FileNotFoundError, IOError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam. "
              "Check that a camera is connected and not in use by another app.")
        sys.exit(1)

    prev_time = time.time()
    fps = 0.0

    print("Starting real-time emotion recognition. Press 'Q' to quit, 'S' to save a screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from webcam. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            faces = predictor.detect_faces(gray)
        except cv2.error as e:
            print(f"WARNING: Face detection error: {e}")
            faces = []

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]

            if face_roi.size == 0:
                continue

            try:
                label, confidence = predictor.predict_emotion(face_roi)
            except Exception as e:
                print(f"WARNING: Prediction failed for a detected face: {e}")
                continue

            display_label = EMOTION_LABELS_DISPLAY.get(label, label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = f"{display_label}: {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 15), (x + tw + 10, y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        curr_time = time.time()
        elapsed = curr_time - prev_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, "Facial Emotion Recognition", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Facial Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved to {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
