import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

# Define image dimensions and batch size
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # You can adjust this based on your computer's memory

# Define the path to your main data directory
# This ensures the script finds your 'data' folder correctly relative to the script's location
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# --- Data Preprocessing and Augmentation for Training Set ---
# Rescales pixels to 0-1 and applies various augmentations to increase data diversity
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values (0-255 to 0-1)
    rotation_range=20, # Randomly rotate images by a max of 20 degrees
    width_shift_range=0.2, # Randomly shift images horizontally
    height_shift_range=0.2, # Randomly shift images vertically
    shear_range=0.2, # Apply shearing transformations
    zoom_range=0.2, # Randomly zoom in on images
    horizontal_flip=True, # Randomly flip images horizontally (important for faces)
    fill_mode='nearest' # Strategy for filling in new pixels created by transformations
)

# --- Data Preprocessing for Validation and Test Sets ---
# Only rescale (normalize) for validation and test sets; no augmentation is applied
# because we want to evaluate the model on real, unaltered data.
validation_test_datagen = ImageDataGenerator(rescale=1./255)

# --- Load Images from Directories using the Generators ---
print("Loading training images...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Resize all images to this dimension
    batch_size=BATCH_SIZE,
    class_mode='binary', # Crucial: Indicates we have two classes (happy-face/sad-face)
    color_mode='rgb', # Assuming color images
    shuffle=True # Shuffle training data for better generalization
)

print("Loading validation images...")
validation_generator = validation_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False # Do not shuffle validation data (for consistent evaluation)
)

print("Loading test images...")
test_generator = validation_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False # Do not shuffle test data (for consistent evaluation)
)

# Print class indices to confirm mapping (e.g., {'happy-face': 0, 'sad-face': 1})
# This shows how the model internally maps your class names to 0s and 1s.
print("\nClass indices:", train_generator.class_indices)

print("Building the CNN model...")
model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25), # Dropout for regularization

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Flattening for Dense layers
    Flatten(),

    # Dense Layers for Classification
    Dense(512, activation='relu'),
    Dropout(0.5), # More dropout for the dense layer
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
# optimizer='adam' is a popular choice
# loss='binary_crossentropy' is used for binary classification with sigmoid output
# metrics=['accuracy'] to monitor performance
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()

# Define callbacks for better training
# EarlyStopping: Stop training if validation loss doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
EPOCHS = 50 # You can adjust the number of training epochs
print(f"\nStarting model training for {EPOCHS} epochs...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per training epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE, # Number of batches per validation epoch
    callbacks=[early_stopping, reduce_lr] # Apply the defined callbacks
)

print("\nModel training complete!")

# Save the trained model
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_emotion_model.h5')
model.save(model_save_path)
print(f"\nModel saved successfully to {model_save_path}")

# Evaluate the model on the test data
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")