# Dealer Video Compliance Assistant (Roboflow + Python)

This project is a small, end to end proof of concept that checks automotive dealer commercials for basic brand compliance using computer vision.

## Live Demo

Streamlit app: https://roboflowdemo-wtv8hsux5c6rwis9jxvyok.streamlit.app/
Code: https://github.com/nrlafata/roboflow_demo

- **Input:** a dealer commercial (MP4) and the expected OEM  
- **Output:** a structured verdict `COMPLIANT` or `NON_COMPLIANT` plus metrics like how often the OEM logo appears and example timestamps

It is built on top of **Roboflow** for data management and model training, and uses **Python + Streamlit** for inference and a lightweight UI.

---

## Problem

OEM and dealer marketing teams often need to verify that video ads follow brand rules, especially around:

- Required OEM logos
- Consistent brand identity
- Avoiding unapproved visuals

Right now this is usually a manual process. Someone watches each commercial, scrubs back and forth, and visually checks whether the correct logo appears often enough. That is slow, subjective, and does not scale when you have many dealers and many ads.

---

## Solution

This project is a prototype "Dealer Video Compliance Assistant".

It:

1. Uses Roboflow to host an object detection model trained on OEM logos, starting with **`mitsubishi_logo`**.
2. Samples frames from a commercial at a fixed rate (for example 1 frame per second) using OpenCV.
3. Sends each frame to the **Roboflow Hosted Inference API**.
4. Aggregates detections across the video and applies simple OEM specific rules, for example:
   - "The logo must appear in at least 30 percent of sampled frames."
   - "If the logo never appears, the spot is non compliant."
5. Returns a structured JSON verdict that can feed into other systems or dashboards.
6. Exposes a **Streamlit web app** so non technical users can upload a commercial, select the OEM, and see the compliance result with metrics.

The same pattern can be extended to support additional OEMs, competitor logos, disclaimers, and more complex Brand Guidelines.

---

## Features

-  Uses a Roboflow trained object detection model (RF-DETR Nano) to detect OEM logos  
-  Samples video frames with OpenCV and aggregates detections over time  
-  Configurable thresholds for logo coverage per OEM  
-  Command line tool for batch or backend use  
-  Streamlit UI for quick demos and non technical users  
-  Returns structured JSON that can be consumed by other services

---

## How it works

### High level flow

1. **Data and model (Roboflow)**  
   - Frames are sampled from real Mitsubishi commercials.  
   - The `mitsubishi_logo` is labeled on those frames.  
   - An RF-DETR Nano model is trained and hosted in Roboflow.  
   - The model is exposed via a Hosted Inference endpoint such as:
     - `https://detect.roboflow.com/my-first-project-9d4yc/2`

2. **Backend logic (Python)**  
   - A video is read with OpenCV and frames are sampled at 1 frame per second.  
   - Each frame is encoded as JPEG and sent to the Roboflow API.  
   - Predictions for each frame are collected.  
   - A simple rule engine checks:
     - Total sampled frames  
     - Frames where `mitsubishi_logo` appears  
     - Ratio of logo frames vs total  
   - A JSON summary is produced with:
     - `status` (COMPLIANT, NON_COMPLIANT, or UNKNOWN)  
     - `reason` (human readable explanation)  
     - `frames_total`, `frames_with_logo`, `logo_frame_ratio`  
     - example timestamps where the logo appears

3. **UI (Streamlit)**  
   - A small Streamlit app wraps the logic.  
   - Users can:
     - upload an MP4  
     - choose the expected OEM from a dropdown  
     - click "Run Compliance Check"  
   - The app displays:
     - a big COMPLIANT or NON COMPLIANT banner  
     - metrics as Streamlit `metric` widgets  
     - a list of timestamps where the logo was detected  
     - the raw JSON result for transparency

---

## Project structure

```text
.
├── app.py            # Streamlit UI
├── check_video.py    # Command line video compliance checker
├── test_image.py     # Optional single image tester for the Roboflow model
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── .gitignore        # Files that should not be committed
└── .env.example      # Example environment variables (no real secrets)



