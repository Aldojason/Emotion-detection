/**
 * script.js
 * Polls the Flask /status endpoint and updates the dashboard UI in real time.
 */

const POLL_INTERVAL_MS = 700;

const el = {
  statusDot: document.getElementById("status-dot"),
  statusText: document.getElementById("status-text"),
  fpsBadge: document.getElementById("fps-badge"),
  emotionValue: document.getElementById("emotion-value"),
  confidenceFill: document.getElementById("confidence-fill"),
  confidenceLabel: document.getElementById("confidence-label"),
  metricModel: document.getElementById("metric-model"),
  metricCamera: document.getElementById("metric-camera"),
  metricFaces: document.getElementById("metric-faces"),
  metricFps: document.getElementById("metric-fps"),
  videoOverlay: document.getElementById("video-overlay"),
  overlayMessage: document.getElementById("overlay-message"),
  errorCard: document.getElementById("error-card"),
  errorText: document.getElementById("error-text"),
};

function setBadge(node, ok) {
  node.classList.remove("good", "bad");
  node.classList.add(ok ? "good" : "bad");
}

async function pollStatus() {
  try {
    const res = await fetch("/status", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateUI(data);
  } catch (err) {
    el.statusDot.classList.remove("ok");
    el.statusDot.classList.add("err");
    el.statusText.textContent = "Disconnected";
  }
}

function updateUI(data) {
  const overallOk = data.camera_available && data.model_available;

  el.statusDot.classList.toggle("ok", overallOk);
  el.statusDot.classList.toggle("err", !overallOk);
  el.statusText.textContent = overallOk ? "System live" : "Attention needed";

  el.fpsBadge.textContent = `${data.fps.toFixed(1)} FPS`;

  el.emotionValue.textContent = data.emotion || "N/A";
  el.confidenceFill.style.width = `${Math.min(data.confidence, 100)}%`;
  el.confidenceLabel.textContent = `${data.confidence.toFixed(1)}% confidence`;

  el.metricModel.textContent = data.model_available ? "Loaded" : "Missing";
  setBadge(el.metricModel, data.model_available);

  el.metricCamera.textContent = data.camera_available ? "Connected" : "Unavailable";
  setBadge(el.metricCamera, data.camera_available);

  el.metricFaces.textContent = data.faces_detected;
  el.metricFps.textContent = `${data.fps.toFixed(1)} FPS`;

  if (!data.camera_available) {
    el.videoOverlay.hidden = false;
    el.overlayMessage.textContent = "Camera unavailable — check connection and permissions.";
  } else {
    el.videoOverlay.hidden = true;
  }

  if (data.error) {
    el.errorCard.hidden = false;
    el.errorText.textContent = data.error;
  } else {
    el.errorCard.hidden = true;
  }
}

pollStatus();
setInterval(pollStatus, POLL_INTERVAL_MS);
