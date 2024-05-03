# # Install Streamlit if not installed
# !pip install streamlit opencv-python tensorflow==2.9.1 protobuf==3.20.*
# !apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = 'cnn_model.h5'  # The path to your trained model
model = load_model(model_path)

# Define constants
IMG_SIZE = (64, 64)  # Image size for model input
THRESHOLD = 0.5  # Threshold for drowsiness detection

# Streamlit UI setup
st.title("Driver Drowsiness Detection System")

# Start the camera capture
camera = cv2.VideoCapture(0)  # Use the default camera (adjust if needed)

# Check if the camera opened successfully
if not camera.isOpened():
    st.error("Could not open webcam")
else:
    st.write("Camera is ready")

# Live stream loop
stframe = st.empty()  # Empty placeholder for live camera stream

while True:
    ret, frame = camera.read()
    if not ret:
        st.warning("Could not read from webcam")
        break

    # Resize and preprocess the frame for the model
    resized_frame = cv2.resize(frame, IMG_SIZE)  # Resize to the input size
    preprocessed_frame = resized_frame / 255.0  # Normalize pixel values
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(preprocessed_frame)
    drowsiness_probability = prediction[0][0]

    # Determine if drowsiness is detected
    is_drowsy = drowsiness_probability > THRESHOLD

    # Overlay text on the frame
    label = "Drowsy" if is_drowsy else "Alert"
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # Red for drowsy, green for alert
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the video frame with prediction result
    stframe.image(frame, channels="BGR", use_column_width=True)

    # Break the loop if 'q' is pressed (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the camera and close any OpenCV windows (cleanup)
camera.release()
cv2.destroyAllWindows()

#Run the project using either of these two commands:
#1
#python -m streamlit run interface.py
#2
#streamlit run interface.py