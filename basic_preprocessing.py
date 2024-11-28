import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd

## 1) Convert raw Pixels to Hounsefield Units ##

def transform_in_hu(img):
    #Check if RescaleSlope and RescaleIntercept exist in the DICOM metadata
    if hasattr(img, 'RescaleSlope') and hasattr(img, 'RescaleIntercept'):
        slope = img.RescaleSlope
        intercept = img.RescaleIntercept
    else:
        slope = 1  # Default value if not present
        intercept = 0  # Default value if not present

    pixel_array = img.pixel_array
    img_hu = pixel_array * slope + intercept
    return img_hu

## 2) Min-Max-Scaling to normalize ##

def normalizer(img):
    minimum = np.min(img)
    maximum = np.max(img)
    img = (img - minimum ) / (maximum - minimum)
    return img

## Finally, Process the pictures ##

# Define the img path!!! -> img path = folder containing your DICOM files

img_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train"

## For later modelling, we will need a directory 2 levels down
output_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train_proc/images"

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Loop through each file in the directory, test the first 100 only!

img_array = []

# How many images do you want to process
k = 3992

for filename in os.listdir(img_path)[:k]:
    # Check if the file is a DICOM file (assuming .dcm extension)
    if filename.endswith('.dcm'):
        image_path = os.path.join(img_path, filename)  # Full file path
        img = dicom.dcmread(image_path)  # Read the DICOM file

    img_hu = transform_in_hu(img)

    ## apply normalization, returns min-max-scaled image
    img_norm = normalizer(img_hu) * 255
    img_norm = img_norm.astype(np.uint8)  # Convert to 8-bit unsigned integer

    # Append the processed image to the array
    img_array.append(img)

    # Save as PNG
    output_file = os.path.join(output_path, filename.replace('.dcm', '.png'))

    # Ensure the file ends with .png
    if not output_file.endswith('.png'):
        output_file += '.png'  # Add .png if it's missing

    img_pil = Image.fromarray(img_norm)
    img_pil.save(output_file)

print(f"✅ Images were processed, converted to .png and saved in {output_path}")

# Convert the list of images to a numpy array
final_array = np.array(img_array)
