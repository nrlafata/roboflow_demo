import os
import json
import base64
import tempfile
import subprocess
from pathlib import Path

import streamlit as st
import requests
from imageio_ffmpeg import get_ffmpeg_exe

# --------------------------
# Safe config loading
# --------------------------
def get_setting(key: str, default: str = "") -> str:
    # st.secrets throws if secrets.toml doesn't exist; handle gracefully
    try:
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)

# --------------------------
# Workflow config
# --------------------------
API_KEY = get_setting("ROBOFLOW_API_KEY")
WORKSPACE_NAME = get_setting("ROBOFLOW_WORKSPACE")
WORKFLOW_ID = get_setting("ROBOFLOW_WORKFLOW_ID")
ROBOFLOW_API_URL = get_setting("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

OEM_CONFIG = {
    "Mitsubishi": {
        "class_name": "mitsubishi_logo",
        "min_logo_ratio": 0.30,
    },
}

CONFIDENCE_THRESHOLD_DEFAULT = 0.5
SAMPLE_FPS_DEFAULT = 1.0


def _find_key_anywhere(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_anywhere(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_anywhere(item, key)
            if found is not None:
                return found
    return None


def call_workflow_frame_jpg_bytes(jpg_bytes: bytes, confidence_threshold: float, target_class: str, debug: bool = False):
    """
    Calls your Roboflow Workflow using plain HTTP.
    No inference_sdk. No cv2.
    """
    if not API_KEY:
        raise RuntimeError("Missing ROBOFLOW_API_KEY.")
    if not WORKSPACE_NAME or not WORKFLOW_ID:
        raise RuntimeError("Missing ROBOFLOW_WORKSPACE or ROBOFLOW_WORKFLOW_ID.")

    b64 = base64.b64encode(jpg_bytes).decode("utf-8")

    url = f"{ROBOFLOW_API_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key": API_KEY,
        "inputs": {
            # must match your workflow input name: "image"
            "image": {"type": "base64", "value": b64}
        },
        "parameters": {
            "confidence_threshold": float(confidence_threshold),
            "target_class": str(target_class),
        }
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    frame_has_logo = _find_key_anywhere(result, "frame_has_logo")
    best_conf = _find_key_anywhere(result, "best_confidence")

    frame_has_logo = bool(frame_has_logo) if frame_has_logo is not None else False
    best_conf = float(best_conf) if best_conf is not None else 0.0

    if debug:
        return frame_has_logo, best_conf, result
    return frame_has_logo, best_conf, None


def sample_frames_ffmpeg(video_path: str, fps: float = 1.0):
    """
    Extract frames using ffmpeg (bundled via imageio-ffmpeg).
    Returns list of (timestamp_seconds, jpg_bytes).
    Timestamps are approximated as frame_index / fps which is fine for demo purposes.
    """
    ffmpeg = get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_pattern = str(Path(tmpdir) / "frame_%06d.jpg")

        # -vf fps=... samples frames
        # -q:v controls jpeg quality (2 is high quality)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            out_pattern
        ]

        subprocess.run(cmd, check=True)

        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        results = []
        for i, frame_path in enumerate(frames):
            ts = i / float(fps)
            jpg_bytes = frame_path.read_bytes()
            results.append((ts, jpg_bytes))

        return results


def evaluate_compliance(frames_eval, min_logo_ratio: float):
    total_frames = len(frames_eval)
    logo_times = [e["time"] for e in frames_eval if e.get("frame_has_logo")]

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
        reason = f"OEM logo appears in only {ratio:.0%} of sampled frames (threshold {min_logo_ratio:.0%})."
    else:
        status = "COMPLIANT"
        reason = f"OEM logo appears in {ratio:.0%} of sampled frames, meeting the {min_logo_ratio:.0%} threshold."

    return {
        "status": status,
        "reason": reason,
        "frames_total": total_frames,
        "frames_with_logo": len(logo_times),
        "logo_frame_ratio": ratio,
        "logo_example_times": logo_times[:10],
    }


def analyze_video(video_path: str, oem_name: str, sample_fps: float, confidence_threshold: float, debug_first_frame: bool = False):
    cfg = OEM_CONFIG[oem_name]
    class_name = cfg["class_name"]
    min_ratio = cfg["min_logo_ratio"]

    frames = sample_frames_ffmpeg(video_path, fps=sample_fps)
    frames_eval = []
    raw_first = None

    for idx, (ts, jpg_bytes) in enumerate(frames, start=1):
        if debug_first_frame and idx == 1:
            frame_has_logo, best_conf, raw = call_workflow_frame_jpg_bytes(
                jpg_bytes,
                confidence_threshold=confidence_threshold,
                target_class=class_name,
                debug=True
            )
            raw_first = raw
        else:
            frame_has_logo, best_conf, _ = call_workflow_frame_jpg_bytes(
                jpg_bytes,
                confidence_threshold=confidence_threshold,
                target_class=class_name,
                debug=False
            )

        frames_eval.append({
            "time": ts,
            "frame_has_logo": frame_has_logo,
            "best_confidence": best_conf
        })

    result = evaluate_compliance(frames_eval, min_ratio)
    return result, raw_first


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Dealer Video Compliance Checker", layout="centered")

st.title("Dealer Video Compliance Checker")
st.write(
    "Upload a dealer commercial, select the expected OEM, and this tool will sample the video, "
    "call a Roboflow Workflow on each sampled frame, and check whether the OEM logo appears frequently enough."
)

with st.expander("Workflow configuration"):
    st.write("These must be set via Streamlit secrets or environment variables.")
    st.code("ROBOFLOW_API_KEY\nROBOFLOW_WORKSPACE\nROBOFLOW_WORKFLOW_ID", language="text")
    st.write(f"Workspace: {WORKSPACE_NAME or '(missing)'}")
    st.write(f"Workflow ID: {WORKFLOW_ID or '(missing)'}")
    st.write(f"API URL: {ROBOFLOW_API_URL}")

if not API_KEY or not WORKSPACE_NAME or not WORKFLOW_ID:
    st.error("Missing Roboflow config. Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID.")
    st.stop()

oem_name = st.selectbox("Expected OEM", list(OEM_CONFIG.keys()))
confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, float(CONFIDENCE_THRESHOLD_DEFAULT), 0.01)
sample_fps = st.slider("Sample FPS", 0.25, 5.0, float(SAMPLE_FPS_DEFAULT), 0.25)
debug_first_frame = st.checkbox("Show raw workflow response for first frame (proof)", value=True)

uploaded_video = st.file_uploader("Upload MP4 video", type=["mp4", "mov", "m4v"], accept_multiple_files=False)
analyze_clicked = st.button("Run Compliance Check", type="primary")

if analyze_clicked:
    if uploaded_video is None:
        st.error("Please upload a video first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            temp_video_path = tmp.name

        with st.spinner("Analyzing video..."):
            try:
                result, raw_first = analyze_video(
                    temp_video_path,
                    oem_name,
                    sample_fps=sample_fps,
                    confidence_threshold=confidence_threshold,
                    debug_first_frame=debug_first_frame
                )
            finally:
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        status = result.get("status", "UNKNOWN")
        reason = result.get("reason", "")

        if status == "COMPLIANT":
            st.success(f"COMPLIANT: {reason}")
        elif status == "NON_COMPLIANT":
            st.error(f"NON-COMPLIANT: {reason}")
        else:
            st.warning(f"{status}: {reason}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sampled Frames", result.get("frames_total", 0))
        col2.metric("Frames with Logo", result.get("frames_with_logo", 0))
        col3.metric("Logo Frame Ratio", f"{result.get('logo_frame_ratio', 0.0):.0%}")

        times = result.get("logo_example_times", [])
        if times:
            st.write("Example timestamps (seconds) where logo was detected:")
            st.write(", ".join(f"{t:.1f}" for t in times))

        if raw_first is not None:
            st.subheader("Workflow raw response (first frame)")
            st.json(raw_first)

        st.subheader("Raw Result JSON")
        st.json(result)
else:
    st.info("Upload a video and click Run Compliance Check to get started.")
