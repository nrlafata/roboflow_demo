import os
import json
import tempfile
from pathlib import Path

import cv2
import requests
import streamlit as st

# --------- CONFIG ---------
MODEL_ID = "my-first-project-9d4yc2"
API_KEY = "YbOtOHJd3JSXN9ARUJrM"

OEM_CONFIG = {
    "Mitsubishi": {
        "class_name": "mitsubishi_logo",
        "min_logo_ratio": 0.30,   # require logo in 30% of frames
    },
    # Later, add more OEMs here with their class names and thresholds
}

CONFIDENCE_THRESHOLD = 0.5  # minimum confidence to count a detection
SAMPLE_FPS = 1.0            # sample 1 frame per second
# --------------------------


# ---------- CORE LOGIC ----------
def call_roboflow_frame(frame_bgr):
    """Encode one frame to JPEG and send to Roboflow as a file upload."""
    success, buffer = cv2.imencode(".jpg", frame_bgr)
    if not success:
        raise RuntimeError("Could not encode frame to JPEG")

    jpg_bytes = buffer.tobytes()

    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {
        "api_key": API_KEY,
        "format": "json",
        "confidence": CONFIDENCE_THRESHOLD,
    }

    files = {
        "file": ("frame.jpg", jpg_bytes, "image/jpeg")
    }

    resp = requests.post(url, params=params, files=files, timeout=30)
    if not resp.ok:
        # this will show up in the Streamlit log if something goes wrong
        print("Roboflow error:", resp.status_code, resp.text)
        resp.raise_for_status()

    data = resp.json()
    return data.get("predictions", [])


def sample_frames(video_path: str, fps: float = 1.0):
    """Return list of (timestamp_seconds, frame_bgr) sampled at `fps`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = fps

    frame_interval = max(int(round(native_fps / fps)), 1)
    frames = []
    frame_idx = 0
    ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frames.append((ts, frame))
            ts += 1.0

        frame_idx += 1

    cap.release()
    return frames


def evaluate_compliance(frames_with_dets, class_name: str, min_logo_ratio: float):
    """Turn per-frame detections into a simple compliance verdict."""
    total_frames = len(frames_with_dets)
    logo_times = []

    for entry in frames_with_dets:
        t = entry["time"]
        classes = [d["class"] for d in entry["detections"]]
        if class_name in classes:
            logo_times.append(t)

    if total_frames == 0:
        return {
            "status": "UNKNOWN",
            "reason": "No frames sampled from video.",
            "frames_total": 0,
        }

    ratio = len(logo_times) / total_frames

    if len(logo_times) == 0:
        status = "NON_COMPLIANT"
        reason = "OEM logo never appears in any sampled frame."
    elif ratio < min_logo_ratio:
        status = "NON_COMPLIANT"
        reason = (
            f"OEM logo appears in only {ratio:.0%} of sampled frames "
            f"(threshold {min_logo_ratio:.0%})."
        )
    else:
        status = "COMPLIANT"
        reason = (
            f"OEM logo appears in {ratio:.0%} of sampled frames, "
            f"meeting the {min_logo_ratio:.0%} threshold."
        )

    return {
        "status": status,
        "reason": reason,
        "frames_total": total_frames,
        "frames_with_logo": len(logo_times),
        "logo_frame_ratio": ratio,
        "logo_example_times": logo_times[:10],
    }


def analyze_video(video_path: str, oem_name: str):
    """End-to-end analysis: sample frames, run Roboflow, evaluate rules."""
    cfg = OEM_CONFIG[oem_name]
    class_name = cfg["class_name"]
    min_ratio = cfg["min_logo_ratio"]

    frames = sample_frames(video_path, fps=SAMPLE_FPS)
    frames_with_dets = []

    for idx, (ts, frame) in enumerate(frames, start=1):
        preds = call_roboflow_frame(frame)
        frames_with_dets.append({"time": ts, "detections": preds})

    result = evaluate_compliance(frames_with_dets, class_name, min_ratio)
    return result
# -------------------------------


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Dealer Video Compliance Checker", layout="centered")

st.title(" Dealer Video Compliance Checker")
st.write(
    "Upload a dealer commercial, select the expected OEM, and this tool will "
    "sample the video, run it through a Roboflow logo detector, and check "
    "whether the OEM logo appears frequently enough to be considered compliant."
)

oem_name = st.selectbox("Expected OEM", list(OEM_CONFIG.keys()))

uploaded_video = st.file_uploader(
    "Upload MP4 video", type=["mp4", "mov", "m4v"], accept_multiple_files=False
)

analyze_clicked = st.button("Run Compliance Check", type="primary")

if analyze_clicked:
    if uploaded_video is None:
        st.error("Please upload a video first.")
    else:
        # Save upload to a temp file so OpenCV can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            temp_video_path = tmp.name

        with st.spinner("Analyzing video... this may take a moment."):
            try:
                result = analyze_video(temp_video_path, oem_name)
            finally:
                # clean up temp file
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        # ---------- Display results ----------
        status = result.get("status", "UNKNOWN")
        reason = result.get("reason", "")
        frames_total = result.get("frames_total", 0)
        frames_with_logo = result.get("frames_with_logo", 0)
        ratio = result.get("logo_frame_ratio", 0.0)
        times = result.get("logo_example_times", [])

        if status == "COMPLIANT":
            st.success(f" COMPLIANT – {reason}")
        elif status == "NON_COMPLIANT":
            st.error(f" NON-COMPLIANT – {reason}")
        else:
            st.warning(f" {status} – {reason}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sampled Frames", frames_total)
        with col2:
            st.metric("Frames with Logo", frames_with_logo)
        with col3:
            st.metric("Logo Frame Ratio", f"{ratio:.0%}")

        if times:
            st.write("Example timestamps (seconds) where logo was detected:")
            st.write(", ".join(f"{t:.1f}" for t in times))

        st.subheader("Raw Result JSON")
        st.json(result)
else:
    st.info("Upload a video and click **Run Compliance Check** to get started.")
# -------------------------------
