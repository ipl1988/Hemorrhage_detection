import pickle
from first_basic_preprocessing import transform_in_hu, normalizer

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pydicom as dicom

# Load the image (replace 'path_to_image' with the actual file path)
img_path = "/home/sebastian/code/ipl1988/raw_data/HU_norm_own_window/ID_9c22cd12c.png"

if img_path.endswith('.dcm'):
    # If the file is a DICOM file
    print("Processing DICOM file...")

    # Load the DICOM image (add your DICOM loading logic here)
    dcm_image = dicom.dcmread(img_path)

    # Apply the transform_in_hu and normalizer functions
    img_hu = transform_in_hu(dcm_image)
    img_norm = normalizer(img_hu)

    # Convert the DICOM image to a format compatible with the model
    img = image.array_to_img(img_norm)  # Ensure it can be used as an image in Keras
    img = img.resize((150, 150))  # Resize to the expected size for your model

if img_path.endswith('.png'):
    # If the file is a PNG file
    print("Processing PNG file...")

    # Load the image
    img = image.load_img(img_path, color_mode="grayscale", target_size=(150, 150))

    # Normalize the image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values

    # Add the batch dimension (model expects batches)
    img_array = np.expand_dims(img_array, axis=0)

    print(f"Image shape after preprocessing: {img_array.shape}")

# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the model file based on the script directory
model_path = os.path.join(script_dir, "..", "model.pkl")  # Goes one level up from the script

# Open the Model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Make the prediction
predictions = loaded_model.predict(img_array)

if predictions[0] < 0.5:
    print("There is no hemorraghe in this image")
    print(f"and I am {(1 - predictions[0][0]):.2f} Percent certain about that")

else:
    print("There is no hemorraghe in this image")
    print(f"and I am {predictions[0][0]:.2f} Percent certain about that")
