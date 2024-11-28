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

## 2) Apply Window ##

def apply_window(img, window_center, window_width):

    # If the field is a MultiValue, extract the first element
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = window_center[0]
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = window_width[0]

    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    img = np.clip(img, lower_bound, upper_bound)
    return img

## 3) Min-Max-Scaling to normalize##

def normalizer(img):
    minimum = np.min(img)
    maximum = np.max(img)
    img = ((img - minimum ) / (minimum - maximum))
    return img

## 4) Process the pictures ##

# Define the input and output path!!!
# Input path = folder containing your DICOM files
# output path = folder to save PNG to

folder_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train"
output_path = "/home/sebastian/code/ipl1988/raw_data/processed_stage_2_train"

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through each file in the directory, test the first 100 only!

img_array = []
k = 100

for filename in os.listdir(folder_path)[0:k]:
    # Check if the file is a DICOM file (assuming .dcm extension)
    if filename.endswith('.dcm'):
        image_path = os.path.join(folder_path, filename)  # Full file path
        img = pydicom.dcmread(image_path)  # Read the DICOM file

    # store temp information needed for the window function
    window_center = img.WindowCenter
    window_width = img.WindowWidth

    ## apply HU transformation, returns transformed img
    img = transform_in_hu(img)

    ## apply unique window, returns windowed img
    img = apply_window(img, window_center, window_width)

    ## apply normalization, returns min-max-scaled image
    img = normalizer(img) * 255
    img = img.astype(np.uint8)  # Convert to 8-bit unsigned integer

    # Append the processed image to the array
    img_array.append(img)

    # Save as PNG
    output_file = os.path.join(output_path, filename.replace('.dcm', '.png'))
    img_pil = Image.fromarray(img)
    img_pil.save(output_file)

final_array = np.array(img_array)
