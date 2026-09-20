# 😷 AI Face Mask Detection

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>

<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>

<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>

<img src="https://img.shields.io/badge/Deep%20Learning-MobileNetV2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="Deep Learning"/>

</p>

An AI-powered **Face Mask Detection System** that uses computer vision and deep learning to detect human faces and classify whether a person is wearing a face mask or not.

The project uses **OpenCV's DNN face detector** for face detection and a **MobileNetV2-based TensorFlow/Keras model** for mask classification. It provides both a **Streamlit web application** and a standalone **real-time video detection system**.

---

## 📌 Project Overview

The system performs two main tasks:

1. **Face Detection**

   * Detects faces from images and video streams.
   * Uses OpenCV's SSD-based deep learning face detector.

2. **Mask Classification**

   * Extracts detected faces.
   * Resizes and preprocesses each face.
   * Uses a trained deep learning model to classify the face as:

     * 😷 **Mask**
     * 🚫 **No Mask**

When a person is detected without a mask, the application displays an alert. The real-time video detector can also provide a voice alert.

---

## ✨ Features

* 😷 AI-based face mask detection
* 👤 Multiple face detection
* 📸 Image upload and analysis
* 📹 Webcam-based detection
* 🎥 Real-time video detection
* 📊 Prediction confidence percentage
* 🚨 No-mask visual alert
* 🔊 Voice alert for no-mask detection
* 🎨 Streamlit web interface
* 🎯 Adjustable face detection confidence threshold
* 🧠 TensorFlow/Keras deep learning model
* 👁️ OpenCV DNN face detection
* 📓 Jupyter/Google Colab training notebooks

---

## 🛠️ Technologies & Tools

<p align="left">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>

<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>

<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>

<img src="https://img.shields.io/badge/MobileNetV2-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="MobileNetV2"/>

<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>

<img src="https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Pillow"/>

<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>

<img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>

<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>

<img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"/>

</p>

---

## 🧠 Model Architecture

The project uses two main models in the detection pipeline.

### 1. Face Detector

The face detector is based on OpenCV's DNN SSD model:

```text
res10_300x300_ssd_iter_140000.caffemodel
```

with its configuration file:

```text
deploy.prototxt
```

The detector identifies faces and returns:

* Face bounding-box coordinates
* Detection confidence

---

### 2. Face Mask Classifier

The mask classification model is stored in:

```text
mask_detector.h5
```

The detected face goes through the following preprocessing pipeline:

```text
Detected Face
      ↓
Extract Face ROI
      ↓
Convert BGR → RGB
      ↓
Resize to 224 × 224
      ↓
Convert to NumPy Array
      ↓
MobileNetV2 Preprocessing
      ↓
Mask Classification Model
      ↓
Mask / No Mask
```

The model produces probabilities for:

```text
Mask
No Mask
```

The class with the higher probability is selected as the prediction.

---

## 🔄 Detection Pipeline

```text
             Input Image / Webcam
                       │
                       ▼
              OpenCV Face Detector
                       │
                       ▼
                  Detect Faces
                       │
                       ▼
                 Extract Face ROI
                       │
                       ▼
                  Resize 224×224
                       │
                       ▼
             MobileNetV2 Preprocessing
                       │
                       ▼
             Mask Classification Model
                       │
                ┌──────┴──────┐
                ▼             ▼
             😷 Mask       🚫 No Mask
                │             │
                └──────┬──────┘
                       ▼
               Display Prediction
                       │
                       ▼
                 No Mask → Alert
```

---

## 📁 Project Structure

```text
AI-Face-Mask-Detection/
│
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── Drive/
│   ├── mask_detector.model
│   └── plot.png
│
├── app.py
├── detect_mask_video.py
├── mask_detector.h5
├── requirements.txt
├── run_app.bat
│
├── test_imports.py
├── test_models.py
├── time_init.py
│
├── train_mask_detector_colab.ipynb
├── train_mask_detector_in_vs_code.ipynb
│
├── plot.png
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/furqanzubair209-cell/AI-Face-Mask-Detection.git
```

Move into the project directory:

```bash
cd AI-Face-Mask-Detection
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
tensorflow
keras
imutils
numpy
opencv-python
matplotlib
scipy
scikit-learn
pillow
streamlit
pyttsx3
```

---

# 🚀 Running the Application

The main application is:

```text
app.py
```

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL that can be opened in your browser.

---

## 📸 Image Detection

The Streamlit application allows you to upload an image containing one or more faces.

The system will:

1. Load the image.
2. Detect faces.
3. Extract each detected face.
4. Classify each face.
5. Display the prediction.
6. Display the confidence percentage.
7. Show an alert if no mask is detected.

Example output:

```text
Mask: 98.52%
```

or:

```text
No Mask: 97.31%
```

---

## 📹 Webcam Detection

The application also supports webcam-based detection.

Select the webcam option in the Streamlit interface and capture an image.

The captured image is then processed by the face detection and mask classification pipeline.

---

# 🎥 Real-Time Video Detection

The project includes:

```text
detect_mask_video.py
```

Run it using:

```bash
python detect_mask_video.py
```

The program will:

1. Start the webcam.
2. Detect faces in real time.
3. Classify each detected face.
4. Draw bounding boxes.
5. Display mask/no-mask predictions.
6. Display prediction confidence.
7. Provide a voice alert when a no-mask face is detected.

Press:

```text
Q
```

to close the video window.

---

# 🪟 Windows Batch File

The project contains:

```text
run_app.bat
```

Windows users can use this file to launch the Streamlit application without manually entering the command.

---

# 📊 Output

The application displays the prediction and confidence around detected faces.

### Mask detected

```text
Mask: 98.52%
```

### No mask detected

```text
No Mask: 97.31%
```

The application can also display:

```text
🚨 ALERT: No Mask Detected!
```

The real-time video detector can additionally provide a voice notification.

---

# 🧪 Testing

The repository includes several testing/helper scripts:

```text
test_imports.py
test_models.py
time_init.py
```

### Test imports

```bash
python test_imports.py
```

### Test models

```bash
python test_models.py
```

These scripts can help verify that the required Python libraries and model files are available.

---

# 📓 Model Training

The repository includes Jupyter notebooks for model development and training.

### Google Colab

```text
train_mask_detector_colab.ipynb
```

This notebook provides a Google Colab-based workflow for training and experimenting with the mask detection model.

### VS Code / Local Environment

```text
train_mask_detector_in_vs_code.ipynb
```

This notebook provides a local development and training workflow.

---

# 📈 Training Visualization

The repository contains:

```text
plot.png
```

which can be used to visualize the model's training performance.

The `Drive` directory also contains model-development files and visualization data.

---

# 🔧 Configuration

The Streamlit application includes a configurable face detection confidence threshold.

The default threshold is:

```text
0.50
```

It can be adjusted using:

```text
Face Detection Confidence
```

A higher threshold makes the detector more selective, while a lower threshold can allow weaker face detections.

---

# 📋 Requirements

Example `requirements.txt`:

```text
tensorflow>=2.5.0
keras
imutils
numpy
opencv-python
matplotlib
scipy
scikit-learn
pillow
streamlit
pyttsx3
```

It is recommended to use a Python environment compatible with the installed TensorFlow version.

---

# ⚠️ Troubleshooting

## Model Files Not Found

Make sure the following files exist:

```text
face_detector/deploy.prototxt
face_detector/res10_300x300_ssd_iter_140000.caffemodel
mask_detector.h5
```

The application expects these files at their specified relative paths.

---

## Webcam Not Working

Make sure:

* Your webcam is connected.
* Camera permissions are enabled.
* Another application is not exclusively using the webcam.
* Python/OpenCV has permission to access the camera.

The standalone video detector uses:

```python
VideoStream(src=0)
```

for the default camera.

---

## TensorFlow / Keras Installation Issues

If you experience dependency problems, create a fresh virtual environment:

```bash
python -m venv venv
```

Activate it and install the dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Important Notes

This project is intended for **educational, demonstration, and computer-vision experimentation purposes**.

Face-mask classification performance can vary depending on:

* Lighting conditions
* Camera quality
* Face orientation
* Distance from the camera
* Occlusion
* Image quality
* Training dataset
* Model limitations

Therefore, predictions should not be considered guaranteed and should not be used as the sole basis for high-stakes decisions.

---

# 🔮 Future Improvements

Possible future improvements include:

* 🎥 Real-time browser video streaming
* 🎯 Improved detection for difficult face angles
* 🧠 Improved model accuracy
* ⚡ GPU acceleration
* 👁️ Face tracking between video frames
* 📊 FPS/performance monitoring
* 📈 Detailed model evaluation metrics
* 🐳 Docker deployment
* ☁️ Cloud deployment
* 📱 Mobile application integration
* 🌐 REST API for mask detection
* 📝 Automatic detection logs
* 🔄 Dataset augmentation
* 🧪 Advanced model evaluation

---

# 👨‍💻 Author

**Muhammad Furqan**

Computer Science Student

---

# 📄 License

This project can be distributed under the **MIT License** if you choose to add an MIT license to the repository.

---

# 🤝 Contributing

Contributions, improvements, and suggestions are welcome.

### 1. Fork the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Face-Mask-Detection.git
```

### 2. Create a feature branch

```bash
git checkout -b feature/improvement
```

### 3. Make your changes

### 4. Commit your changes

```bash
git add .
git commit -m "Add improvement"
```

### 5. Push your branch

```bash
git push origin feature/improvement
```

Then open a Pull Request.

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📌 Keywords

```text
Face Mask Detection
Face Detection
Computer Vision
Deep Learning
Machine Learning
Artificial Intelligence
TensorFlow
Keras
MobileNetV2
OpenCV
Python
Streamlit
CNN
Image Classification
Real-Time Detection
Webcam Detection
```
