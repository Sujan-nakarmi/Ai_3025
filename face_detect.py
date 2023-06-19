import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import os

data_dir = "dataset_images"
label_file = "label.txt"

labels = []

# Read labels from label.txt file
with open(label_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip().split(" ")[1]
        labels.append(label)

data = []
labels_encoded = []

# Load images and encode labels
for label_idx, folder in enumerate(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        data.append(img)
        labels_encoded.append(label_idx)

data = np.array(data)
labels_encoded = np.array(labels_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("facedetection.h5")
