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
face_classifier = cv2.CascadeClassifier('C:/Users/sriteja kattekola/Documents/mini project/haarcascade_frontalface_default.xml')

def load_emotion_model():
    try:
        model = load_model('C:/Users/sriteja kattekola/Documents/mini project/emotion_detection_model_22000.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

classifier = load_emotion_model()

class_labels = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Streamlit app
st.title("Emotion Detector")
run = st.checkbox('Run')

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the first camera device

if cap.isOpened():
    st.write("Camera opened successfully")
else:
    st.error("Failed to open camera")

frame_window = st.image([])

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video feed.")
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
