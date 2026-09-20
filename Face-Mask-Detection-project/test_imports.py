import os
import cv2
print("Checking files...")
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

print(f"Prototxt: {os.path.exists(prototxtPath)}")
print(f"Weights: {os.path.exists(weightsPath)}")

print("Loading face detector...")
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
print("Face detector loaded.")

print("Now attempting TF import...")
import tensorflow as tf
print(f"TF Version: {tf.__version__}")
print("Test complete.")
