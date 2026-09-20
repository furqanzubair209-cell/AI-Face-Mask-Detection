import os
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

print("Checking models...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(prototxtPath):
    print(f"MISSING: {prototxtPath}")
else:
    print(f"FOUND: {prototxtPath}")

if not os.path.exists(weightsPath):
    print(f"MISSING: {weightsPath}")
else:
    print(f"FOUND: {weightsPath}")

try:
    print("Loading face detector...")
    face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("Face detector loaded successfully.")
except Exception as e:
    print(f"Error loading face detector: {e}")

model_path = "mask_detector.h5"
if not os.path.exists(model_path):
    print(f"MISSING: {model_path}")
else:
    print(f"FOUND: {model_path}")

try:
    print("Loading mask detector model (this might take a moment)...")
    mask_model = load_model(model_path)
    print("Mask detector loaded successfully.")
except Exception as e:
    print(f"Error loading mask detector: {e}")

print("Test complete.")
