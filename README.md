# 🎓 Classroom Attentiveness Detection using YOLOv9s & Streamlit

A real-time **student attentiveness monitoring system** using **YOLOv9s** and **Streamlit**. This project detects students in a classroom video (or webcam feed) and classifies their attention state as:

* **Attentive** ✅
* **Looking Down** 👀
* **Distracted** ⚠️

This is a **proof-of-concept** app suitable for small classroom setups and research purposes.

---

## 🚀 Features

* **Real-time detection** of students from webcam or uploaded video.
* **YOLOv9s** for fast and accurate person detection.
* **Rule-based attention classifier** (can be replaced with a trained CNN for better accuracy).
* **Streamlit UI** for easy deployment and visualization.
* **Lightweight & CPU-friendly** (works without GPU for small videos).

---

## 🛠 Tech Stack

* Python 3.11+
* [Streamlit](https://streamlit.io/)
* [Ultralytics YOLOv9](https://github.com/ultralytics/ultralytics)
* OpenCV (`opencv-python-headless`)
* NumPy

---
## ⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/classroom-attentiveness.git
cd classroom-attentiveness
```

2. Create a virtual environment and activate:

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure `yolov9s.pt` is in the project root.

---

## 🖥 Usage

### Run Streamlit app:

```bash
streamlit run app.py
```

### Options in the app:

1. **Upload Video:** Upload a classroom video (`.mp4`, `.avi`, `.mov`) to analyze.
2. **Webcam:** Use your local webcam for real-time detection.

The app will display bounding boxes around detected students and label their attention state.

---

## 🧠 How it Works

1. **YOLOv9s** detects all people in the frame.

2. Each detected student bounding box is passed to a **simple attention classifier**.

3. The attention classifier uses **bounding box aspect ratio** as a proxy for head orientation:

   * Ratio < 0.75 → *Looking Down*
   * Ratio 0.75–1.2 → *Attentive*
   * Ratio > 1.2 → *Distracted*

4. The frame is annotated and displayed in Streamlit.

> ⚠️ For production, replace the rule-based classifier with a trained CNN or head-pose detection model for better accuracy.

---

## 📈 Next Steps / Improvements

* Train a **small CNN classifier** to detect attentiveness more accurately.
* Integrate **head-pose estimation** (MediaPipe or OpenCV DNN) for better gaze detection.
* Generate **attention analytics** (per student, per session).
* Support **multi-camera classrooms**.
* Deploy on **edge devices** with optimized ONNX/TensorRT models.

---

## 🔒 Privacy & Ethics

* Obtain **consent from students** or guardians before recording.
* Prefer **local processing** without storing raw video.
* Ensure **fairness and accuracy** across diverse student groups.

---

## 📄 License

MIT License. See `LICENSE` file for details.

---

## 📌 References

* [YOLOv9 Ultralytics](https://github.com/ultralytics/ultralytics)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [OpenCV Documentation](https://opencv.org/)

---
