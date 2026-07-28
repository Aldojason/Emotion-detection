"""
streamlit_app.py
Facial Emotion Recognition — Streamlit version with continuous live video.

Uses streamlit-webrtc so the browser's webcam feed is streamed to the
server over WebRTC, processed frame-by-frame (face detection + emotion
prediction drawn on top), and streamed back — giving a real-time
experience similar to the original Flask MJPEG version, but running
entirely inside Streamlit's execution model.
"""

import os
import sys
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))
from utils import (
    EMOTION_LABELS, EMOTION_LABELS_DISPLAY, MODEL_PATH,
    preprocess_face, load_face_cascade
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="\U0001F4F7",
    layout="wide",
)

# Minimal dark theme touch-ups (Streamlit's own theme covers most of this;
# this just tightens spacing and styles the metric cards a bit).
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: #171c22;
            border: 1px solid #232a32;
            border-radius: 10px;
            padding: 12px 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Facial Emotion Recognition")
st.caption("Real-time CNN inference \u00b7 FER2013 \u00b7 streamlit-webrtc")


# ---------------------------------------------------------------------------
# Cached resource loading — runs once per server process, shared across
# all visitor sessions (st.cache_resource, not st.cache_data, because a
# Keras model / cv2 cascade are not serializable "data").
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading emotion model...")
def get_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading face detector...")
def get_face_cascade():
    try:
        return load_face_cascade()
    except IOError:
        return None


model = get_model()
face_cascade = get_face_cascade()

if model is None:
    st.error(
        "Trained model not found at `models/emotion_model.h5`. "
        "Run `python training/train.py` first, then make sure the file "
        "is committed to your repo before deploying."
    )
    st.stop()

if face_cascade is None:
    st.error("Could not load the OpenCV face detector.")
    st.stop()


# ---------------------------------------------------------------------------
# Video processor — runs in a background thread managed by streamlit-webrtc.
# It must not touch Streamlit UI elements directly (different thread), so
# it just stores the latest results behind a lock; the main script polls
# that via a periodic autorefresh.
# ---------------------------------------------------------------------------
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade
        self.lock = threading.Lock()
        self.latest = {"faces": [], "count": 0}
        self.frame_count = 0
        self.cached_faces = []
        self.cached_results = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        # Run detection and prediction every 3 frames to prevent WebRTC timeout & save CPU
        if self.frame_count % 3 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
                )
            except cv2.error:
                faces = []

            results = []
            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                try:
                    processed = preprocess_face(roi)
                    # Directly calling model is much faster than model.predict()
                    predictions = self.model(processed, training=False).numpy()[0]
                    idx = int(np.argmax(predictions))
                    label = EMOTION_LABELS[idx]
                    confidence = float(predictions[idx]) * 100.0
                except Exception:
                    continue

                display_label = EMOTION_LABELS_DISPLAY.get(label, label)
                results.append({
                    "rect": (x, y, w, h),
                    "label": display_label,
                    "confidence": confidence
                })

            self.cached_faces = faces
            self.cached_results = results

            # Update latest info for the Streamlit UI metrics panel
            with self.lock:
                self.latest = {
                    "faces": [{"label": r["label"], "confidence": round(r["confidence"], 1)} for r in results],
                    "count": len(faces)
                }

        # Draw cached results on skipped frames to keep overlay smooth
        for r in self.cached_results:
            (x, y, w, h) = r["rect"]
            display_label = r["label"]
            confidence = r["confidence"]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{display_label}: {confidence:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, y - th - 12), (x + tw + 8, y), (0, 255, 0), -1)
            cv2.putText(img, text, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if len(self.cached_faces) == 0:
            cv2.putText(img, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_latest(self):
        with self.lock:
            return dict(self.latest)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_video, col_data = st.columns([2, 1])

# Public STUN server for NAT traversal. If viewers on restrictive networks
# (corporate/school Wi-Fi) can't connect, you can add your TURN server credentials
# to the Streamlit secrets panel (turn_url, turn_username, turn_credential).
ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

if "turn_url" in st.secrets:
    ice_servers.append({
        "urls": st.secrets["turn_url"],
        "username": st.secrets.get("turn_username", ""),
        "credential": st.secrets.get("turn_credential", ""),
    })

RTC_CONFIGURATION = RTCConfiguration({"iceServers": ice_servers})

with col_video:
    ctx = webrtc_streamer(
        key="emotion-recognition",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_data:
    st.subheader("Live status")

    if ctx.state.playing:
        # Trigger a rerun every 700ms so the metrics below reflect the
        # video processor's latest results without the user clicking
        # anything. This only runs while the stream is actually playing.
        st_autorefresh(interval=700, key="metrics-refresh")

        result = ctx.video_processor.get_latest() if ctx.video_processor else {"faces": [], "count": 0}
        faces = result.get("faces", [])

        if faces:
            primary = faces[0]
            st.metric("Detected emotion", primary["label"], f"{primary['confidence']}% confidence")
        else:
            st.metric("Detected emotion", "N/A", "0% confidence")

        st.metric("Faces in frame", result.get("count", 0))

        if len(faces) > 1:
            st.caption("All detected faces:")
            for i, f in enumerate(faces, start=1):
                st.write(f"{i}. {f['label']} — {f['confidence']}%")
    else:
        st.info("Click **Start** on the video panel and allow camera access to begin.")

    st.divider()
    st.caption("Emotion classes")
    st.write(", ".join(EMOTION_LABELS_DISPLAY.values()))

st.divider()
st.caption(
    "Facial Emotion Recognition System \u00b7 CNN trained on FER2013 \u00b7 "
    "TensorFlow + OpenCV + Streamlit + streamlit-webrtc"
)