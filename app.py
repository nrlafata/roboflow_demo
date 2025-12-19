import os
import json
import tempfile
import streamlit as st

# --------------------------
# Workflow config
# --------------------------
API_KEY = st.secrets.get("ROBOFLOW_API_KEY", os.environ.get("ROBOFLOW_API_KEY", ""))
WORKSPACE_NAME = st.secrets.get("ROBOFLOW_WORKSPACE", os.environ.get("ROBOFLOW_WORKSPACE", ""))
WORKFLOW_ID = st.secrets.get("ROBOFLOW_WORKFLOW_ID", os.environ.get("ROBOFLOW_WORKFLOW_ID", ""))

# Workflows run on serverless
ROBOFLOW_API_URL = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

OEM_CONFIG = {
    "Mitsubishi": {
        "class_name": "mitsubishi_logo",
        "min_logo_ratio": 0.30,
    },
}

CONFIDENCE_THRESHOLD_DEFAULT = 0.5
SAMPLE_FPS_DEFAULT = 1.0


@st.cache_resource
def get_rf_client():
    if not API_KEY:
        raise RuntimeError("Missing ROBOFLOW_API_KEY. Set it in Streamlit secrets or an env var.")
    return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=API_KEY)


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


def call_workflow_frame(frame_bgr, confidence_threshold, target_class, debug=False):
   
    if not WORKSPACE_NAME or not WORKFLOW_ID:
        raise RuntimeError("Missing ROBOFLOW_WORKSPACE or ROBOFLOW_WORKFLOW_ID.")

    client = get_rf_client()

    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": frame_bgr},  # must match your Workflow Input name: "image"
        parameters={
            "confidence_threshold": confidence_threshold,
            "target_class": target_class,
        },
        use_cache=False,  # helpful while iterating
    )

    frame_has_logo = _find_key_anywhere(result, "frame_has_logo")
    best_conf = _find_key_anywhere(result, "best_confidence")

    frame_has_logo = bool(frame_has_logo) if frame_has_logo is not None else False
    best_conf = float(best_conf) if best_conf is not None else 0.0

    if debug:
        return frame_has_logo, best_conf, result
    return frame_has_logo, best_conf, None


def sample_frames(video_path: str, fps: float = 1.0):
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


def analyze_video(video_path: str, oem_name: str, sample_fps: float, confidence_threshold: float, debug_first_frame=False):
    cfg = OEM_CONFIG[oem_name]
    class_name = cfg["class_name"]
    min_ratio = cfg["min_logo_ratio"]

    frames = sample_frames(video_path, fps=sample_fps)
    frames_eval = []

    raw_first = None

    for idx, (ts, frame) in enumerate(frames, start=1):
        if debug_first_frame and idx == 1:
            frame_has_logo, best_conf, raw = call_workflow_frame(
                frame,
                confidence_threshold=confidence_threshold,
                target_class=class_name,
                debug=True
            )
            raw_first = raw
        else:
            frame_has_logo, best_conf, _ = call_workflow_frame(
                frame,
                confidence_threshold=confidence_threshold,
                target_class=class_name
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
    st.code(
        "ROBOFLOW_API_KEY\nROBOFLOW_WORKSPACE\nROBOFLOW_WORKFLOW_ID",
        language="text"
    )
    st.write(f"Workspace: {WORKSPACE_NAME or '(missing)'}")
    st.write(f"Workflow ID: {WORKFLOW_ID or '(missing)'}")

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
        with col1:
            st.metric("Total Sampled Frames", result.get("frames_total", 0))
        with col2:
            st.metric("Frames with Logo", result.get("frames_with_logo", 0))
        with col3:
            st.metric("Logo Frame Ratio", f"{result.get('logo_frame_ratio', 0.0):.0%}")

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

