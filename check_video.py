import cv2
import requests
import json
from pathlib import Path

# ---------- CONFIG ----------
VIDEO_PATH = "FS1212FSM_FiveStarMitsubishi-YearEndSalesEvent_30s.mp4"

MODEL_ID = "my-first-project-9d4yc/2"
API_KEY = "YbOtOHJd3JSXN9ARUJrM"

OEM_LOGO_CLASS = "mitsubishi_logo"  # must match your Roboflow class
CONFIDENCE_THRESHOLD = 0.5          # only count detections above this
MIN_LOGO_RATIO = 0.3                # require logo in 30% of frames
# -----------------------------


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
        # helpful debug
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


def evaluate_compliance(frames_with_dets):
    """Turn per-frame detections into a simple compliance verdict."""
    total_frames = len(frames_with_dets)
    logo_times = []

    for entry in frames_with_dets:
        t = entry["time"]
        classes = [d["class"] for d in entry["detections"]]
        if OEM_LOGO_CLASS in classes:
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
    elif ratio < MIN_LOGO_RATIO:
        status = "NON_COMPLIANT"
        reason = (
            f"OEM logo appears in only {ratio:.0%} of frames "
            f"(threshold {MIN_LOGO_RATIO:.0%})."
        )
    else:
        status = "COMPLIANT"
        reason = (
            f"OEM logo appears in {ratio:.0%} of frames, "
            f"meeting the {MIN_LOGO_RATIO:.0%} threshold."
        )

    return {
        "status": status,
        "reason": reason,
        "frames_total": total_frames,
        "frames_with_logo": len(logo_times),
        "logo_frame_ratio": ratio,
        "logo_example_times": logo_times[:5],
    }


def main():
    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    print(f"Sampling frames from: {video_path}")
    frames = sample_frames(str(video_path), fps=1.0)
    print(f"Sampled {len(frames)} frames. Sending to Roboflow...")

    frames_with_dets = []

    for idx, (ts, frame) in enumerate(frames, start=1):
        preds = call_roboflow_frame(frame)
        frames_with_dets.append({"time": ts, "detections": preds})

        logo_present = any(p["class"] == OEM_LOGO_CLASS for p in preds)
        print(
            f"[{idx}/{len(frames)}] t={ts:4.1f}s  "
            f"detections={len(preds)}  logo_present={logo_present}"
        )

    result = evaluate_compliance(frames_with_dets)

    print("\n=== COMPLIANCE RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
