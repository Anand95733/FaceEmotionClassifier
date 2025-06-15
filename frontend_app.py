from flask import Flask, render_template, request, jsonify
import requests # To make HTTP requests to your FastAPI backend
import os

app = Flask(__name__)

# Define the URL for your FastAPI backend
# Make sure this matches where your FastAPI app is running
FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# --- Home Page Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Prediction Route ---
@app.route('/predict_flask', methods=['POST'])
def predict_flask():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if image_file:
        # Prepare the file to be sent to FastAPI
        # requests expects a tuple: ('filename', file_object, 'content_type')
        files = {'file': (image_file.filename, image_file.stream, image_file.mimetype)}

        try:
            # Send the image to the FastAPI /predict endpoint
            response = requests.post(f"{FASTAPI_BASE_URL}/predict", files=files)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Get the JSON response from FastAPI
            fastapi_response_data = response.json()

            # Return FastAPI's response directly to the frontend
            return jsonify(fastapi_response_data)

        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Could not connect to the FastAPI server. Please ensure it is running at " + FASTAPI_BASE_URL}), 500
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Error during API request: {e}"}), 500
        except Exception as e:
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

    return jsonify({"error": "Something went wrong with the file."}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)