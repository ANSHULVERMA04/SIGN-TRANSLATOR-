import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2


X = np.load("X.npy")  # pre processed images 
Y = np.load("Y.npy")  # labels 

# Normalize images
X = X / 255.0


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(set(Y)), activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X, Y, epochs=10, validation_split=0.2)


model.save("sign_language_model.h5")
