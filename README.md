# AI-Based Image Classification Web Application: Face Emotion Classifier

## Project Description
This project implements a robust AI-based web application designed to classify human facial emotions. It utilizes a pre-trained deep learning model to accurately predict emotions from static images uploaded by the user and also offers an interactive real-time emotion prediction feature via webcam. The application showcases a complete full-stack AI/ML system, with a responsive frontend built using Flask and a powerful backend powered by FastAPI for efficient model inference and API handling.

## Features
*   **Image Prediction:**
    *   Upload any image containing a human face.
    *   The backend API processes the image using the emotion classification model.
    *   The predicted emotion (e.g., "Happy," "Sad") is displayed along with a confidence score.
    *   Probabilities for all recognized emotion classes are also shown, providing a detailed breakdown.
*   **Live Webcam Prediction:**
    *   Activate your device's webcam directly from the web interface.
    *   Frames are continuously streamed to the FastAPI backend for real-time processing.
    *   The system performs on-the-fly face detection using OpenCV Haar Cascades.
    *   For each detected face, an emotion prediction is made and the result is displayed live on the frontend, offering an interactive experience.
*   **Prediction History:**
    *   All successful image predictions are automatically saved locally in your browser's storage.
    *   A dedicated "Prediction History" page allows you to review past predictions, including the original uploaded image, the predicted emotion, confidence, and the timestamp of the prediction.
    *   This provides a convenient way to track and review previous interactions with the image prediction feature.
*   **Responsive User Interface:**
    *   A clean, intuitive, and modern web interface designed for ease of use.
    *   Built with Flask, complemented by HTML for structure, CSS for styling, and JavaScript for dynamic interactions.
    *   The layout adapts well to different screen sizes, from desktop monitors to mobile devices.

## Technical Stack
*   **ML Framework:** TensorFlow/Keras for building, training, and deploying the deep learning emotion classification model.
*   **API Backend:** FastAPI, a modern, fast (high-performance) web framework for building RESTful APIs with Python 3.7+, based on standard Python type hints. It automatically generates interactive API documentation (Swagger UI).
*   **Frontend:** Flask, a lightweight Python web framework used for building the user interface and handling client-side interactions and routing.
*   **Face Detection:** OpenCV (Open Source Computer Vision Library) utilized with Haar Cascade Classifiers for efficient and robust face detection in images and video streams.
*   **Data Format:** Supports common image formats like JPG and PNG for input.
*   **Deployment (Local):** Configured for development and deployment on a local machine (localhost).

## Project Structure

```
FaceEmotionClassifier/
├── app.py                         # FastAPI backend application
├── train_model.py                 # Script to train the ML model
├── frontend_app.py                # Flask frontend application
├── detect_emotion.py              # (Optional: If specific detection logic is separated)
├── requirements.txt               # Python package dependencies
├── face_emotion_model.h5          # Trained Keras model (generated after training)
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── README.md                      # This documentation file
├── data/                          # Dataset directory
│   ├── train/                     # Training images
│   │   ├── happy-face/
│   │   └── sad-face/
│   ├── validation/                # Validation images
│   │   ├── happy-face/
│   │   └── sad-face/
│   └── test/                      # Test images
│       ├── happy-face/
│       └── sad-face/
├── static/                        # Frontend static assets (CSS, JS, images)
│   ├── godhaar_logo.png
│   ├── profile-pic.png
│   ├── script.js
│   └── style.css
└── templates/                     # Flask HTML templates
└── index.html
```

## Setup Instructions

Follow these detailed steps to get the "Face Emotion Classifier" web application up and running on your local machine.

### 1. Prerequisites
Before you begin, ensure you have the following installed:
*   **Python 3.8 or higher:** You can download it from [python.org](https://www.python.org/downloads/).
*   **`pip`:** Python's package installer, which usually comes bundled with Python.

### 2. Clone the Repository (or navigate to your project folder)
If your project is hosted on a version control system like Git:

1.  Open your terminal or command prompt.
2.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/FaceEmotionClassifier.git](https://github.com/yourusername/FaceEmotionClassifier.git) # Replace with your actual repository URL
    cd FaceEmotionClassifier
    ```
    If you manually copied the files, simply open your terminal or command prompt and navigate to the FaceEmotionClassifier root directory.

### 3. Create and Populate the `data` Directory

1.  Create a directory named `data` in the project root.
2.  Inside the `data` directory, create three subdirectories: `train`, `validation`, and `test`.
3.  Inside each of these subdirectories (`train`, `validation`, `test`), create two subdirectories: `happy-face` and `sad-face`.
4.  Place your happy and sad face images into their respective folders. Ensure you have a minimum of 200 images per class for training.

    Your directory structure should look like this:

    ```
    data/
    ├── train/
    │   ├── happy-face/
    │   │   └── [happy face images]
    │   └── sad-face/
    │       └── [sad face images]
    ├── validation/
    │   ├── happy-face/
    │   │   └── [happy face images]
    │   └── sad-face/
    │       └── [sad face images]
    └── test/
        ├── happy-face/
        │   └── [happy face images]
        └── sad-face/
            └── [sad face images]
    ```

### 4. Obtain `haarcascade_frontalface_default.xml`

This XML file is crucial for real-time face detection using OpenCV.

1.  Download the file: You can reliably find it in the official OpenCV GitHub repository. Right-click and choose "Save link as..." or "Save target as..." from this link: [https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml)
2.  Place the file: Save the downloaded `haarcascade_frontalface_default.xml` file directly into your `FaceEmotionClassifier` project root directory (the same folder where your `app.py` and `train_model.py` are located).

### 5. Set Up Python Virtual Environment

1.  Navigate to your `FaceEmotionClassifier` project directory in your terminal or PowerShell.

    ```bash
    cd C:\Users\ANAND KUMAR E\OneDrive\Desktop\Nxt-Assignments\FaceEmotionClassifier
    ```

2.  Create a virtual environment named `venv`:

    ```bash
    python -m venv venv
    ```

3.  Activate the virtual environment:

    *   For Windows (PowerShell):
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    *   For macOS / Linux / Git Bash:
        ```bash
        source venv/bin/activate
        ```

    You should see `(venv)` at the beginning of your terminal prompt, indicating the environment is active.

### 6. Install Dependencies

With your virtual environment activated, install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This command will install FastAPI, Uvicorn, TensorFlow, Flask, OpenCV, and other necessary libraries for your project.

### 7. Train the Emotion Classification Model (Optional)

If you don't have the `face_emotion_model.h5` file from a previous training session, or if you wish to retrain it with updated data:

1.  Ensure your `data` folder (with `train`, `validation`, `test` subfolders) is fully populated with images.
2.  Run the training script:
    ```bash
    python train_model.py
    ```

This script will train the model, evaluate its performance, and save the trained model as `face_emotion_model.h5` in your project root directory.

## How to Use the App

The application consists of two separate servers: the FastAPI backend and the Flask frontend. You will need two separate terminal windows to run them.

### Step 1: Start the FastAPI Backend

1.  Open a **NEW** Terminal or PowerShell window.
2.  Navigate to your project directory:
    ```bash
    cd C:\Users\ANAND KUMAR E\OneDrive\Desktop\Nxt-Assignments\FaceEmotionClassifier
    ```
3.  Activate your virtual environment:
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    *   macOS / Linux / Git Bash: `source venv/bin/activate`
4.  Run the FastAPI application:
    ```bash
    uvicorn app:app --reload
    ```

    The backend will typically run on `http://127.0.0.1:8000`. Keep this terminal window open and running.

### Step 2: Start the Flask Frontend

1.  Open **another NEW** Terminal or PowerShell window.
2.  Navigate to your project directory:
    ```bash
    cd C:\Users\ANAND KUMAR E\OneDrive\Desktop\Nxt-Assignments\FaceEmotionClassifier
    ```
3.  Activate your virtual environment (again):
    *   Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
    *   macOS / Linux / Git Bash: `source venv/bin/activate`
4.  Run the Flask application:
    ```bash
    python frontend_app.py
    ```

    The frontend will typically run on `http://127.0.0.1:5000`. Keep this terminal window open and running.

### Step 3: Access the Web Application

1.  Open your web browser.
2.  Go to: `http://127.0.0.1:5000` or `http://localhost:5000`
3.  You should now see the Face Emotion Classifier web interface.

### Usage Flow

1.  **Image Prediction:**
    *   Click on the "Image Prediction" tab.
    *   Upload an image containing a human face using the "Choose File" button.
    *   Click the "Predict" button.
    *   The predicted emotion and confidence score will be displayed below the button. You'll also see probabilities for all recognized emotion classes.
    *   Your prediction will be saved in the "Prediction History".
2.  **Live Webcam Prediction:**
    *   Click on the "Webcam Prediction" tab.
    *   Click the "Start Webcam" button to activate your webcam. You may need to grant the website permission to access your webcam.
    *   The application will detect faces in the webcam feed and display the predicted emotion in real-time.
    *   Click the "Stop Webcam" button to stop the webcam feed.
3.  **Prediction History:**
    *   Click on the "Prediction History" tab.
    *   You will see a list of your previous image predictions, including the image, predicted emotion, confidence, and timestamp.

## Troubleshooting

*   **"No Face Detected" Error:**
    *   Ensure `haarcascade_frontalface_default.xml` is in the correct project root directory.
    *   Check the FastAPI backend terminal for any errors related to `haarcascade_frontalface_default.xml` when the server starts.
    *   Consider adjusting `minNeighbors` (e.g., to `3` or `2`) or `minSize` (e.g., to `(30,30)`) parameters in `app.py`'s `detectMultiScale` function to make face detection less strict. Remember to save `app.py` and restart the FastAPI server after changes.
*   **"Emotion model could not be loaded" Error:**
    *   Ensure `face_emotion_model.h5` exists in the project root directory. If it's missing, you need to run `python train_model.py` after populating your `data/` folder.
*   **Dependency Errors:**
    *   Make sure your virtual environment is activated and you have run `pip install -r requirements.txt` successfully.
*   **Port Conflicts:** If either server fails to start because a port is already in use, you might need to change the port in `app.py` (for FastAPI) or `frontend_app.py` (for Flask), or identify and stop the process using that port.

## Images

<table style="width:100%">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/6364a47b-19f5-4552-9aee-a653bf870992" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/703363b6-0bc6-48e1-8a5e-d63dad1b9521" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/595de2e3-953b-4828-a5ea-643a80d6639f" width="200"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/58bf6ac1-3ccb-4b6d-992a-79fdcecd4f30" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/1ef36cf7-2ddd-4c34-9866-52fdc6ba3a81" width="200"></td>
    <td><img src="https://github.com/user-attachments/assets/37564c6f-a371-4f0d-a69b-06a31a817ae5" width="200"></td>
  </tr>
</table>
