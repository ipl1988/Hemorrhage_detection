from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the dataset
img_test_dir_path = "/home/sebastian/code/ipl1988/test_data_dont_touch"
labels = pd.read_csv("/home/sebastian/code/ipl1988/test_data_labels.csv")['Label']


# Use image_dataset_from_directory
test_dataset = image_dataset_from_directory(
    directory=img_test_dir_path,
    labels=list(labels),
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),
    color_mode="grayscale",  # Keep it grayscale for loading
    shuffle=False
)

# Apply the conversion (you could do this during preprocessing)
test_dataset = test_dataset.map(lambda x, y: (tf.image.grayscale_to_rgb(x), y))

for images, labels in test_dataset.take(1):  # Take one batch
    print(images.shape)  # Should print (batch_size, 150, 150, 3)
    print(labels.shape)  # Should match your label shape

# Preprocessing for MobileNetV2 (this includes normalization specific to ImageNet)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Apply the preprocessing function
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))

# Load pre-trained MobileNetV2 model with ImageNet weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add classification head (Fully connected layer) on top
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (adjust if you have more classes)
])

# Freeze the base model layers (optional)
base_model.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
