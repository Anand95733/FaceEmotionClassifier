import cv2
import numpy as np
from tensorflow.keras.models import load_model # Used to load your saved model
from tensorflow.keras.preprocessing.image import img_to_array # To convert image to array
import os

# --- Configuration ---
# Define image dimensions (must match what the model was trained on)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define the path to your saved model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_emotion_model.h5')

# Load the trained model
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'face_emotion_model.h5' exists in the project root directory.")
    exit() # Exit if model loading fails

# Load OpenCV's pre-trained Haar Cascade for face detection
# This XML file should be available with your OpenCV installation or downloaded.
# You might need to adjust this path if cv2.data.haarcascades is not correct on your system.
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(face_cascade_path)

if face_classifier.empty():
    print(f"Error: Face cascade XML file not loaded. Check path: {face_cascade_path}")
    print("You might need to download 'haarcascade_frontalface_default.xml' and place it in your project folder,")
    print("or verify your OpenCV installation provides it.")
    exit() # Exit if cascade not loaded

# Define emotion labels (must match your class indices: {'happy-face': 0, 'sad-face': 1})
# This assumes 'happy-face' is 0 and 'sad-face' is 1 as per your training output.
# If your output showed {'sad-face': 0, 'happy-face': 1}, you'd swap them.
emotion_labels = ['Happy', 'Sad']

# --- Start Webcam Capture ---
cap = cv2.VideoCapture(0) # 0 for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit() # Exit if webcam can't be opened

print("\nWebcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for face detection (Haar cascades work on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

        # Extract the face region of interest (ROI)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Preprocess the face ROI for the model
        # Resize to the target dimensions your model expects
        # Convert to RGB (if it's not already, and model expects RGB)
        # Normalize pixel values (0-255 to 0-1)
        # Add batch dimension (model expects a batch of images)
        try:
            face_roi = cv2.resize(roi_color, (IMG_WIDTH, IMG_HEIGHT))
            face_roi = img_to_array(face_roi) # Convert to NumPy array
            face_roi = np.expand_dims(face_roi, axis=0) # Add batch dimension
            face_roi = face_roi / 255.0 # Normalize pixel values
        except cv2.error as e:
            print(f"Error resizing face ROI: {e}. Skipping this face.")
            continue # Skip to the next detected face

        # Make a prediction
        prediction = model.predict(face_roi)[0][0] # Get the single probability value

        # Determine the emotion based on the prediction
        # Sigmoid output is probability. If > 0.5, it's the 2nd class (Sad), else 1st class (Happy)
        emotion_index = 1 if prediction > 0.5 else 0
        emotion_text = emotion_labels[emotion_index]

        # Display the prediction on the frame
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Emotion Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()