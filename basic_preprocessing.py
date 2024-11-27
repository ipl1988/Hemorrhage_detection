import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd

## 1) Convert raw Pixels to Hounsefield Units ##

def transform_in_hu(img):
    slope = img.RescaleSlope
    intercept = img.RescaleIntercept
    img = img.pixel_array * slope + intercept
    return img

## 3) Min-Max-Scaling to normalize ##

def normalizer(img):
    minimum = np.min(img)
    maximum = np.max(img)
    img = (img - minimum ) / (maximum - minimum)
    return img

## 4) Process the pictures ##

# Define the input and output path!!!
# Input path = folder containing your DICOM files
# output path = folder to save PNG to

img_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train"

def make_output_path(img_path):
    return os.path.join(img_path, "images_stage_2_train_png")

output_path = make_output_path(img_path)

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through each file in the directory, test the first 100 only!

img_array = []
k = 10

for filename in os.listdir(img_path)[:k]:
    # Check if the file is a DICOM file (assuming .dcm extension)
    if filename.endswith('.dcm'):
        image_path = os.path.join(img_path, filename)  # Full file path
        img = dicom.dcmread(image_path)  # Read the DICOM file

    img = transform_in_hu(img)

    ## apply normalization, returns min-max-scaled image
    img = normalizer(img) * 255
    img = img.astype(np.uint8)  # Convert to 8-bit unsigned integer

    # Append the processed image to the array
    img_array.append(img)

    # Save as PNG
    output_file = os.path.join(output_path, filename.replace('.dcm', '.png'))
    img_pil = Image.fromarray(img)
    img_pil.save(output_file)

print(f"✅ Images were processed, converted to .png and saved in {output_path}")

final_array = np.array(img_array)
