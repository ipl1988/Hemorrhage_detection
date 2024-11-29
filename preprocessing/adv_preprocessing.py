import pydicom as dicom
import matplotlib.pyplot as plt
import os
import shutil
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import pandas as pd
import random

# Load variables from the .env file
load_dotenv()
output_path = os.getenv("output_path")
img_path = os.getenv("img_path")

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

## 2) Apply Window ##

def apply_window(img, window_center, window_width):
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    img = np.clip(img, lower_bound, upper_bound)
    return img

## 3) Min-Max-Scaling to normalize ##

def normalizer(img):
    minimum = np.min(img)
    maximum = np.max(img)
    img = (img - minimum ) / (maximum - minimum)
    return img

## Finally, Process the pictures ##

if __name__ == "__main__":

    # Ensure the output directory exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # Create the directory fresh
    os.makedirs(output_path)


    def process_images(k=100, window_center=40, window_width=80):

        all_files = [f for f in os.listdir(img_path) if f.endswith('.dcm')]
        selected_files = random.sample(all_files, min(k, len(all_files)))

        img_array = []

        for filename in selected_files:
            # Check if the file is a DICOM file (assuming .dcm extension)
            image_path = os.path.join(img_path, filename)  # Full file path
            img = dicom.dcmread(image_path)  # Read the DICOM file

            # transform image in HU
            img_hu = transform_in_hu(img)

            # Apply Brain window of center=40 and width=80
            img_window = apply_window(img_hu, window_center, window_width)

            ## apply normalization, returns min-max-scaled image
            img_norm = normalizer(img_window) * 255
            img_norm = img_norm.astype(np.uint8)  # Convert to 8-bit unsigned integer

            # Append the processed image to the array
            img_array.append(img_norm)

            # Save as PNG
            output_file = os.path.join(output_path, filename.replace('.dcm', '.png'))

            # Ensure the file ends with .png
            if not output_file.endswith('.png'):
                output_file += '.png'  # Add .png if it's missing

            img_pil = Image.fromarray(img_norm)
            img_pil.save(output_file)


        # Convert the list of images to a numpy array
        final_array = np.array(img_array)

        print(f"✅ {k} Images were processed, converted to .png and saved in {output_path}")
        return final_array

    process_images(k = 500)
