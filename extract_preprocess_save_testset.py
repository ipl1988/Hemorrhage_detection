from preprocessing.first_basic_preprocessing import transform_in_hu, normalizer

import pydicom as dicom
import os
import numpy as np
from PIL import Image
import pandas as pd
import random
from PIL import Image


img_dir_full = "/home/sebastian/code/ipl1988/testordner/stage_2_train_full"
img_dir_4000 = "/home/sebastian/code/ipl1988/raw_data/stage_2_train"
output_path = "/home/sebastian/code/ipl1988/testordner/stage_2_test_proc/images"
csv_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train.csv"


# Ensures the output directory exists, creates it otherwise
if not os.path.exists(output_path):
    os.makedirs(output_path)


### helper functions ###

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

def normalizer(img):
    minimum = np.min(img)
    maximum = np.max(img)
    img = (img - minimum ) / (maximum - minimum)
    return img

## Process the pictures ##
def process_images(k):

    # List all DICOM files in the directory
    all_files = [f for f in os.listdir(img_dir_full) if f.endswith('.dcm')]

    # Get the list of already processed files in the img_dir_4000
    existing_files = [f for f in os.listdir(img_dir_4000) if f.endswith('.png')]

    # Filter out files already present in img_dir_4000
    new_files = [f for f in all_files if f.replace('.dcm', '.png') not in existing_files]

    # Randomly sample k files (or all new files if k > len(new_files))
    selected_files = random.sample(new_files, min(k, len(new_files)))

    img_array = []
    for filename in selected_files:
        # Check if the file is a DICOM file (assuming .dcm extension)
        image_path = os.path.join(img_dir_full, filename)  # Full file path
        img = dicom.dcmread(image_path)  # Read the DICOM file
        img_hu = transform_in_hu(img)
        ## apply normalization, returns min-max-scaled image
        img_norm = normalizer(img_hu) * 255
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
    print(f"✅ Images were processed, converted to .png and saved in {output_path}")
    return final_array

# Draw 1000 images randomly and process them
process_images(10)



#### Extract the labels for the 1000 test_images ####
df = pd.read_csv(csv_path)

# Extract the pure Image ID and save in new column base_ID
df['base_ID'] = df['ID'].str.extract(r'^(ID_[^_]+)')

# ## !!! I want to look up the label of the ID + _any !!! ###
# df_filtered = df[df['ID'] == key + "_any"]['Label']
# true_value = filtered_df['Label']

# convert to dictioniary in the form id: labellist e.g. [0,1,0,1,0]
grouped_data = df.groupby('base_ID')['Label'].apply(list).to_dict()

img_list = os.listdir(output_path)

labels=[]
actual_values = []

for file_name in img_list:
    # Remove the .png extension (or any other extension)
    key = os.path.splitext(file_name)[0]

    if key in grouped_data:
        # Get the last label from the list for the current file
        label = grouped_data[key][-1]
        true_value = df[df['ID'] == key + "_any"]['Label']

        # Append the label to both lists
        labels.append(label)
        actual_values.append(true_value)
    else:
        # Handle the case where the key is not found in the grouped data
        print(f"Warning: {key} not found in grouped_data")

print("Start Cross-Checking if the labels correspond correctly")
# Assert that labels match this_list
if labels == actual_values:
    print("The labels match this_list!")
else:
    raise AssertionError("The labels do not match this_list!")
