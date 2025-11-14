import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Load YOLOv9s model
# -------------------------------
@st.cache_resource
def load_model():
    # Ensure yolov9s.pt is in the same folder or give full path
    model = YOLO("yolov9s.pt")
    return model

model = load_model()

# -------------------------------
# Simple attention classifier (rule-based)
# -------------------------------
def classify_attention(box):
    """
    Rule-based attention classification based on bounding box aspect ratio.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    ratio = w / h if h != 0 else 1

    if ratio < 0.75:
        return "Looking Down"
    elif 0.75 <= ratio <= 1.2:
        return "Attentive"
    else:
        return "Distracted"

# -------------------------------
# Draw bounding box & label
# -------------------------------
def draw_result(frame, box, label):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) if label == "Attentive" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# -------------------------------
# Process a single frame
# -------------------------------
def process_frame(frame, model):
    # Ensure frame is uint8 and contiguous for YOLO
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            if int(cls) == 0:  # person class
                attention = classify_attention(box)
                frame = draw_result(frame, box, attention)
    return frame

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Student Attentiveness Detection - YOLOv9s")
st.write("Upload a classroom video or use webcam for real-time analysis.")

option = st.radio("Input Source:", ["Upload Video", "Webcam"], horizontal=True)

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name

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

elif option == "Webcam":
    st.write("Allow webcam access and click 'Start Webcam'.")
    run = st.checkbox("Start Webcam")

    if run:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (720, 480))
                frame = process_frame(frame, model)
                stframe.image(frame, channels="BGR")
                run = st.checkbox("Start Webcam", value=True)
            cap.release()
