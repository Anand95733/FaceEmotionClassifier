from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os
import cv2 # NEW: Import OpenCV

# --- Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_emotion_model.h5')
# NEW: Path to Haar Cascade classifier
face_cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')

emotion_labels = ['Happy', 'Sad']

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Face Emotion Classifier API",
    description="API for classifying happy and sad emotions from face images.",
    version="1.0.0",
)

# Add CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:5000",
    "http://127.0.0.1",
    "http://127.0.0.1:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the Trained Model and Face Cascade ---
model = None
face_cascade = None # NEW: Initialize face_cascade as None

@app.on_event("startup")
async def load_ml_model_and_cascade():
    """Load the machine learning model and face cascade classifier when the FastAPI application starts up."""
    global model, face_cascade
    try:
        model = load_model(model_path)
        print(f"Emotion model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print("Please ensure 'face_emotion_model.h5' exists and is a valid Keras model.")
        raise HTTPException(status_code=500, detail="Emotion model could not be loaded. Server cannot start.")

    try:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            raise ValueError("Haar Cascade classifier XML file not loaded.")
        print(f"Face cascade classifier loaded successfully from {face_cascade_path}")
    except Exception as e:
        print(f"Error loading face cascade classifier: {e}")
        print("Please ensure 'haarcascade_frontalface_default.xml' exists in the project root directory and is valid.")
        raise HTTPException(status_code=500, detail="Face cascade classifier could not be loaded. Server cannot start.")


# --- Test Endpoint ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Face Emotion Classifier API! Visit /docs for API documentation."}

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Convert PIL Image to OpenCV format (NumPy array)
        cv_image = np.array(pil_image)
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        # NEW: Face Detection
        # detectMultiScale(image, scaleFactor, minNeighbors)
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        # --- UPDATED: minNeighbors changed from 5 to 3 for broader detection ---
        faces = face_cascade.detectMultiScale(cv_image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

        if len(faces) == 0:
            # No face detected
            return {
                "filename": file.filename,
                "predicted_class": "No Face Detected",
                "confidence": 0.0,
                "all_probabilities": {label: 0.0 for label in emotion_labels},
                "detail": "No face detected in the image."
            }

        # For simplicity, take the first detected face (assuming one main face per image for this app)
        x, y, w, h = faces[0]
        
        # Crop the face from the original color image
        face_crop_cv = cv_image[y:y+h, x:x+w]
        
        # Convert the cropped face back to PIL Image, then resize for the model
        face_pil = Image.fromarray(cv2.cvtColor(face_crop_cv, cv2.COLOR_BGR2RGB))
        image_resized = face_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert resized image to numpy array for model prediction
        image_array = img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        raw_prob_sad = model.predict(image_array)[0][0]

        prob_sad = float(raw_prob_sad)
        prob_happy = float(1 - raw_prob_sad)

        if prob_sad > prob_happy:
            predicted_class = emotion_labels[1]  # 'Sad'
            confidence = prob_sad
        else:
            predicted_class = emotion_labels[0]  # 'Happy'
            confidence = prob_happy

        all_probabilities = {
            emotion_labels[0]: round(prob_happy * 100, 2),
            emotion_labels[1]: round(prob_sad * 100, 2)
        }

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image or making prediction: {e}")