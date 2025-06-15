// --- Global Elements ---
const navLinks = document.querySelectorAll('.nav-links a');
const sections = document.querySelectorAll('.main-content-wrapper > .container');
const uploadForm = document.getElementById('upload-form');
const imageUpload = document.getElementById('image-upload');
const uploadedImage = document.getElementById('uploaded-image');
const resultDiv = document.getElementById('result');
const historyGallery = document.getElementById('history-gallery');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const noHistoryMessage = document.getElementById('no-history-message');

// --- NEW WEBCAM ELEMENTS ---
const webcamVideo = document.getElementById('webcam-feed');
const webcamCanvas = document.getElementById('webcam-canvas');
const webcamContext = webcamCanvas.getContext('2d');
const startWebcamBtn = document.getElementById('start-webcam-btn');
const stopWebcamBtn = document.getElementById('stop-webcam-btn');
const webcamStatus = document.getElementById('webcam-status');
const liveEmotion = document.getElementById('live-emotion');
const liveConfidence = document.getElementById('live-confidence');

// --- Webcam Variables ---
let currentStream; // To hold the MediaStream object
let animationFrameId; // To hold the ID for requestAnimationFrame loop
let isPredictingLive = false; // Flag to control the prediction loop

// --- Configuration ---
const CONFIDENCE_THRESHOLD = 60; // Set your desired confidence threshold (e.g., 60% or 70%)
const PREDICTION_INTERVAL_MS = 200; // How often to send frames for prediction (e.g., 200ms = 5 predictions/sec)

// --- Section Visibility Logic ---
function showSection(sectionId) {
    // Deactivate all sections
    sections.forEach(section => {
        section.classList.remove('active-section');
        section.style.opacity = '0';
        section.style.pointerEvents = 'none';
        section.style.display = 'none'; // Explicitly hide

        // --- NEW: Stop webcam if leaving the live prediction section ---
        if (section.id === 'live-prediction-section') {
            stopWebcam();
        }
    });

    // Deactivate all nav links
    navLinks.forEach(link => {
        link.classList.remove('active');
    });

    // Activate the target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.style.display = 'block'; // Show before transition
        setTimeout(() => { // Small delay to allow display to take effect before opacity transition
            targetSection.classList.add('active-section');
            targetSection.style.opacity = '1';
            targetSection.style.pointerEvents = 'auto';
        }, 10); // Very short delay

        // Activate the corresponding navigation link
        const correspondingNavLink = document.querySelector(`.nav-links a[data-section="${sectionId}"]`);
        if (correspondingNavLink) {
            correspondingNavLink.classList.add('active');
        }

        // Handle specific section behaviors
        if (sectionId === 'prediction-history-section') {
            loadAndDisplayHistory(); // Load history when history section is active
        } else if (sectionId === 'live-prediction-section') {
            // No automatic start here, user clicks button
            webcamStatus.textContent = 'Webcam Ready';
            liveEmotion.textContent = 'N/A';
            liveConfidence.textContent = 'N/A';
            startWebcamBtn.disabled = false;
            stopWebcamBtn.disabled = true;
        }


        // Scroll to top of the visible section after a short delay for transition
        setTimeout(() => {
            targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    } else {
        console.warn(`Section with ID '${sectionId}' not found.`);
        // Fallback to home if invalid sectionId
        showSection('main-prediction-section');
    }
}

// --- Navigation Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Initial display based on URL hash or default
    const initialHash = window.location.hash.substring(1);
    if (initialHash && document.getElementById(initialHash)) {
        showSection(initialHash);
    } else {
        showSection('main-prediction-section');
        history.replaceState(null, null, '#main-prediction-section'); // Set default hash
    }

    // Add event listeners to navigation links (using data-section)
    navLinks.forEach(link => {
        if (link.hasAttribute('data-section')) {
            link.addEventListener('click', function(event) {
                event.preventDefault();
                const targetId = this.getAttribute('data-section');
                showSection(targetId);
                history.pushState(null, null, '#' + targetId);
            });
        }
    });

    // Listen for browser back/forward buttons (hash change)
    window.addEventListener('hashchange', () => {
        const currentHash = window.location.hash.substring(1);
        if (currentHash && document.getElementById(currentHash)) {
            showSection(currentHash);
        } else {
            showSection('main-prediction-section');
        }
    });

    // Initial check for no history
    checkNoHistoryMessage();
});


// --- Image Upload and Prediction Logic ---
uploadedImage.style.display = 'none'; // Hide initially

imageUpload.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage.src = e.target.result;
            uploadedImage.style.display = 'block'; // Show image preview
            resultDiv.innerHTML = ''; // Clear previous results
            resultDiv.classList.remove('success', 'error'); // Clear previous styling
        };
        reader.readAsDataURL(file);
    } else {
        uploadedImage.src = '';
        uploadedImage.style.display = 'none';
        resultDiv.innerHTML = '';
        resultDiv.classList.remove('success', 'error');
    }
});

uploadForm.addEventListener('submit', async function(event) {
    event.preventDefault(); // Prevent default form submission

    const file = imageUpload.files[0];
    if (!file) {
        resultDiv.classList.add('error');
        resultDiv.innerHTML = 'Please select an image first.';
        return;
    }

    resultDiv.classList.remove('success', 'error');
    resultDiv.innerHTML = 'Predicting... <span class="loading-spinner"></span>'; // Show loading spinner

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Prediction Result:", data);

        if (data.predicted_class && data.confidence !== undefined && data.all_probabilities) {
            resultDiv.classList.remove('error');
            resultDiv.classList.add('success');

            // Display predicted emotion with color
            resultDiv.innerHTML = `Predicted: <strong class="predicted-emotion ${data.predicted_class}">${data.predicted_class}</strong> with ${data.confidence.toFixed(2)}% confidence.`;

            let probabilitiesHtml = '<div class="probability-details">';
            probabilitiesHtml += '<h3>All Probabilities:</h3>';
            probabilitiesHtml += '<ul class="probability-list">';
            for (const emotion in data.all_probabilities) {
                probabilitiesHtml += `<li>${emotion}: <span>${data.all_probabilities[emotion].toFixed(2)}%</span></li>`;
            }
            probabilitiesHtml += '</ul>';
            probabilitiesHtml += '</div>';

            resultDiv.innerHTML += probabilitiesHtml;

            // Low Confidence Feedback
            if (data.confidence < CONFIDENCE_THRESHOLD) {
                const feedbackMessage = document.createElement('p');
                feedbackMessage.classList.add('feedback-message');
                feedbackMessage.innerHTML = `<br>
                    <small>The model had low confidence in this prediction. For better results, try a clearer image of a single face, or provide feedback if you think this is a misclassification.</small>
                    <button class="feedback-button">Give Feedback</button>
                `;
                resultDiv.appendChild(feedbackMessage);

                feedbackMessage.querySelector('.feedback-button').addEventListener('click', () => {
                    alert('Thank you for your feedback! This feature can be expanded to collect data for model improvement.');
                    console.log('Feedback submitted for:', {
                        image: uploadedImage.src,
                        predicted: data.predicted_class,
                        confidence: data.confidence,
                        true_label: 'user_correction_needed'
                    });
                });
            }

            // Save prediction to history
            const fileReaderForHistory = new FileReader();
            fileReaderForHistory.onload = function(e) {
                const imageDataUrl = e.target.result;
                savePredictionToHistory(imageDataUrl, data);
            };
            fileReaderForHistory.readAsDataURL(file);

        } else {
            resultDiv.classList.remove('success');
            resultDiv.classList.add('error');
            resultDiv.innerHTML = `Error: Could not retrieve full prediction details.`;
            console.error("Incomplete data received:", data);
        }

    } catch (error) {
        console.error('Prediction failed:', error);
        resultDiv.classList.remove('success');
        resultDiv.classList.add('error');
        resultDiv.innerHTML = `Prediction failed: ${error.message || 'An unknown error occurred.'}`;
    }
});


// --- NEW: Live Webcam Prediction Logic ---

startWebcamBtn.addEventListener('click', startWebcam);
stopWebcamBtn.addEventListener('click', stopWebcam);

async function startWebcam() {
    webcamStatus.textContent = 'Starting webcam...';
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcamVideo.srcObject = currentStream;
        webcamVideo.play();

        webcamVideo.onloadedmetadata = () => {
            webcamCanvas.width = webcamVideo.videoWidth;
            webcamCanvas.height = webcamVideo.videoHeight;
            webcamStatus.textContent = 'Webcam On. Predicting...';
            startWebcamBtn.disabled = true;
            stopWebcamBtn.disabled = false;
            isPredictingLive = true;
            predictWebcamFrame(); // Start the prediction loop
        };
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        webcamStatus.textContent = `Error: ${err.name || 'Failed to access webcam.'}`;
        liveEmotion.textContent = 'N/A';
        liveConfidence.textContent = 'N/A';
        startWebcamBtn.disabled = false;
        stopWebcamBtn.disabled = true;
    }
}

function stopWebcam() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    isPredictingLive = false;
    webcamVideo.srcObject = null;
    webcamContext.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height); // Clear canvas
    webcamStatus.textContent = 'Webcam Off.';
    liveEmotion.textContent = 'N/A';
    liveConfidence.textContent = 'N/A';
    startWebcamBtn.disabled = false;
    stopWebcamBtn.disabled = true;
}

let lastPredictionTime = 0;

async function predictWebcamFrame(currentTime) {
    if (!isPredictingLive) {
        return; // Stop if the flag is false
    }

    // Control prediction rate
    if (currentTime - lastPredictionTime < PREDICTION_INTERVAL_MS) {
        animationFrameId = requestAnimationFrame(predictWebcamFrame);
        return;
    }
    lastPredictionTime = currentTime;

    if (webcamVideo.readyState === webcamVideo.HAVE_ENOUGH_DATA) {
        webcamContext.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        // Convert canvas image to Blob for sending
        webcamCanvas.toBlob(async (blob) => {
            if (blob) {
                const formData = new FormData();
                formData.append('file', blob, 'webcam_frame.png'); // Naming the blob

                try {
                    const response = await fetch('http://127.0.0.1:8000/predict', {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    if (data.predicted_class && data.confidence !== undefined) {
                        liveEmotion.innerHTML = `<span class="predicted-emotion ${data.predicted_class}">${data.predicted_class}</span>`;
                        liveConfidence.textContent = `${data.confidence.toFixed(2)}%`;
                        webcamStatus.textContent = 'Predicting...';
                    } else if (data.detail === "No face detected in the image.") {
                        liveEmotion.textContent = 'No Face';
                        liveConfidence.textContent = 'N/A';
                        webcamStatus.textContent = 'Looking for face...';
                    } else {
                        console.warn("Unexpected prediction response:", data);
                        liveEmotion.textContent = 'Error';
                        liveConfidence.textContent = 'N/A';
                        webcamStatus.textContent = 'Prediction error!';
                    }
                } catch (error) {
                    console.error('Live prediction failed:', error);
                    liveEmotion.textContent = 'Error';
                    liveConfidence.textContent = 'N/A';
                    webcamStatus.textContent = `Connection error: ${error.message}`;
                    // Optionally stop webcam on persistent error
                    // stopWebcam();
                }
            }
        }, 'image/png'); // Specify image format
    }

    animationFrameId = requestAnimationFrame(predictWebcamFrame);
}


// --- Prediction History Logic ---

const HISTORY_KEY = 'faceEmotionPredictions';

function savePredictionToHistory(imageDataUrl, predictionData) {
    let history = JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];

    const historyItem = {
        timestamp: new Date().toISOString(),
        image_data_url: imageDataUrl,
        predicted_class: predictionData.predicted_class,
        confidence: predictionData.confidence,
        all_probabilities: predictionData.all_probabilities
    };

    history.unshift(historyItem);

    if (history.length > 20) {
        history = history.slice(0, 20);
    }

    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    loadAndDisplayHistory();
    checkNoHistoryMessage();
}

function loadAndDisplayHistory() {
    let history = JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    historyGallery.innerHTML = '';

    if (history.length === 0) {
        noHistoryMessage.style.display = 'block';
    } else {
        noHistoryMessage.style.display = 'none';
        history.forEach(item => {
            const historyItemDiv = document.createElement('div');
            historyItemDiv.classList.add('history-item');

            const img = document.createElement('img');
            img.src = item.image_data_url;
            img.alt = 'Predicted Image';
            img.classList.add('history-thumbnail');
            historyItemDiv.appendChild(img);

            const detailsDiv = document.createElement('div');
            detailsDiv.classList.add('history-details');

            const timestamp = new Date(item.timestamp).toLocaleString();
            const confidence = item.confidence !== undefined ? item.confidence.toFixed(2) : 'N/A';

            detailsDiv.innerHTML = `
                <p><strong>Predicted:</strong> <span class="predicted-emotion ${item.predicted_class}">${item.predicted_class}</span></p>
                <p><strong>Confidence:</strong> ${confidence}%</p>
                <p class="timestamp">${timestamp}</p>
            `;

            if (item.all_probabilities) {
                let probListHtml = '<ul class="history-probability-list">';
                for (const emotion in item.all_probabilities) {
                    probListHtml += `<li>${emotion}: <span>${item.all_probabilities[emotion].toFixed(2)}%</span></li>`;
                }
                probListHtml += '</ul>';
                detailsDiv.innerHTML += probListHtml;
            }

            historyItemDiv.appendChild(detailsDiv);
            historyGallery.appendChild(historyItemDiv);
        });
    }
    checkNoHistoryMessage();
}

function clearHistory() {
    if (confirm('Are you sure you want to clear your prediction history?')) {
        localStorage.removeItem(HISTORY_KEY);
        loadAndDisplayHistory();
        checkNoHistoryMessage();
    }
}

function checkNoHistoryMessage() {
    const history = JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    if (history.length === 0) {
        noHistoryMessage.style.display = 'block';
    } else {
        noHistoryMessage.style.display = 'none';
    }
}

// Event Listener for Clear History Button
clearHistoryBtn.addEventListener('click', clearHistory);