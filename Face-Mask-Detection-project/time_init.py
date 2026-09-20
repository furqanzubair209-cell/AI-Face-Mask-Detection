import time
import os

start_total = time.time()
print("Starting imports...")
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

import_time = time.time()
print(f"Imports took: {import_time - start_total:.2f} seconds")

prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

face_net_time = time.time()
print(f"Face net load took: {face_net_time - import_time:.2f} seconds")

model_path = "mask_detector.h5"
mask_model = load_model(model_path)

mask_model_time = time.time()
print(f"Mask model load took: {mask_model_time - face_net_time:.2f} seconds")
print(f"Total initialization took: {mask_model_time - start_total:.2f} seconds")
