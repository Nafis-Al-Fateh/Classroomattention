import streamlit as st
import torch
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# -------------------------------
# Load YOLOv9s model
# -------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov9s.pt")
    return model

model = load_model()

# -------------------------------
# Simple attentiveness heuristic
# -------------------------------
def classify_attention(face_box):
    """
    Basic rule-based attention classifier.
    face_box = [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = face_box
    w = x2 - x1
    h = y2 - y1

    # Ratio of width to height gives pose clues
    ratio = w / h

    if ratio < 0.75:
        return "Looking Down"
    elif 0.75 <= ratio <= 1.20:
        return "Attentive"
    else:
        return "Distracted"


# -------------------------------
# Draw bounding boxes
# -------------------------------
def draw_result(frame, box, label):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


# -------------------------------
# Process a frame
# -------------------------------
def process_frame(frame, model):
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # class 0 = person
                attention = classify_attention(box)
                frame = draw_result(frame, box, attention)

    return frame


# -------------------------------
# MAIN STREAMLIT UI
# -------------------------------
st.title("🎓 Student Attentiveness Detection - YOLOv9s")
st.write("Real-time or uploaded video stream analysis using YOLOv9s")

option = st.radio("Choose Input Source:",
                  ["Upload Video", "Webcam"],
                  horizontal=True)

# -------------------------------
# Upload Video Mode
# -------------------------------
if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a classroom video", type=["mp4", "mov", "avi"])

    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

        stframe = st.empty()

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (720, 480))
            frame = process_frame(frame, model)
            stframe.image(frame, channels="BGR")

        cap.release()

# -------------------------------
# Webcam Mode
# -------------------------------
elif option == "Webcam":
    st.write("Allow webcam access to begin.")

    run = st.checkbox("Start Webcam")

    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame = cv2.resize(frame, (720, 480))
            frame = process_frame(frame, model)

            stframe.image(frame, channels="BGR")
            run = st.checkbox("Start Webcam", value=True)

        cap.release()
