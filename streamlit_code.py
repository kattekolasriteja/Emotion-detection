# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:54:54 2024

@author: sriteja kattekola
"""

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained models
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_emotion_model():
    try:
        model = load_model('emotion_detection_model_22000.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

classifier = load_emotion_model()

class_labels = ['angry', 'disgusted','fear','happy','neutral','sad','surprised']

# Streamlit app
st.title("Emotion Detector")
run = st.checkbox('Run')

# Try different indices to find the correct camera
for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"Cannot open camera at index {i}")
    else:
        print(f"Camera found at index {i}")
        break


frame_window = st.image([])

# Continue with capturing frames from the correct camera index
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video feed.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert the frame to RGB and display it in the Streamlit app
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()
# cv2.destroyAllWindows()  # This line can be safely removed
