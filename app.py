import os
import sys
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))
from utils import (
    EMOTION_LABELS, EMOTION_LABELS_DISPLAY, MODEL_PATH,
    preprocess_face, load_face_cascade
)

app = Flask(__name__)

class AppState:
    
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_available = False
        self.model_available = False
        self.current_emotion = "N/A"
        self.current_confidence = 0.0
        self.fps = 0.0
        self.error_message = None
        self.faces_detected = 0

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def snapshot(self):
        with self.lock:
            return {
                "camera_available": self.camera_available,
                "model_available": self.model_available,
                "emotion": self.current_emotion,
                "confidence": round(self.current_confidence, 1),
                "fps": round(self.fps, 1),
                "faces_detected": self.faces_detected,
                "error": self.error_message,
            }


state = AppState()

model = None
face_cascade = None
camera = None


def load_resources():
    """Loads the trained model and face detector at startup, if available."""
    global model, face_cascade

    if os.path.isfile(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH)
            state.update(model_available=True)
            print("Emotion model loaded successfully.")
        except Exception as e:
            state.update(model_available=False, error_message=f"Model load error: {e}")
            print(f"ERROR loading model: {e}")
    else:
        state.update(
            model_available=False,
            error_message="Trained model not found. Please run training/train.py first.",
        )
        print("WARNING: emotion_model.h5 not found. Run training/train.py to train it.")

    try:
        face_cascade = load_face_cascade()
    except IOError as e:
        state.update(error_message=f"Face detector error: {e}")
        print(f"ERROR loading face cascade: {e}")


def get_camera():
    """Lazily initializes the webcam capture device."""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
            state.update(
                camera_available=False,
                error_message="Camera unavailable. Check that a webcam is connected "
                              "and not in use by another application.",
            )
        else:
            state.update(camera_available=True, error_message=None)
    return camera


def generate_frames():
    """Generator that yields MJPEG-encoded frames with emotion overlays."""
    cam = get_camera()
    prev_time = time.time()

    if cam is None:
        # Yield a single placeholder frame indicating camera failure
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera unavailable", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        ret, buffer = cv2.imencode(".jpg", placeholder)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        return

    while True:
        success, frame = cam.read()
        if not success:
            state.update(error_message="Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        if face_cascade is not None:
            try:
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                )
            except cv2.error as e:
                state.update(error_message=f"Face detection error: {e}")

        detected_count = len(faces)
        best_label, best_conf = "N/A", 0.0

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            if face_roi.size == 0:
                continue

            label, confidence = "N/A", 0.0
            if model is not None:
                try:
                    processed = preprocess_face(face_roi)
                    predictions = model.predict(processed, verbose=0)[0]
                    idx = int(np.argmax(predictions))
                    label = EMOTION_LABELS[idx]
                    confidence = float(predictions[idx]) * 100.0
                    best_label, best_conf = label, confidence
                except Exception as e:
                    state.update(error_message=f"Prediction error: {e}")

            display_label = EMOTION_LABELS_DISPLAY.get(label, label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{display_label}: {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if detected_count == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        prev_time = curr_time

        state.update(
            current_emotion=EMOTION_LABELS_DISPLAY.get(best_label, best_label),
            current_confidence=best_conf,
            fps=fps,
            faces_detected=detected_count,
        )

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Renders the main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Streams the live annotated webcam feed as MJPEG."""
    return Response(generate_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    """Returns the current system status as JSON (polled by the frontend)."""
    return jsonify(state.snapshot())


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
