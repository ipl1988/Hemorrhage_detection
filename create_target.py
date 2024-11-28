import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd
import random

## Specify these paths first!!! ##

output_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train_proc/images"
csv_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train.csv"

# Read target CSV file

df = pd.read_csv(csv_path)

##### Extract the Image ID

# Assuming the ID column contains both the image ID (e.g., ID_12cadc6af) and the type of hemorrhage (e.g., epidural, intraparenchymal),
# we need to extract just the base ID. The base ID is the part before the first underscore.
# This can be done using the str.extract() method to grab everything up to the first underscore

df['base_ID'] = df['ID'].str.extract(r'^(ID_[^_]+)')

#### Group the Labels by the Base Image ID

# Now, we can group the DataFrame by this base ID and collect all the corresponding Label values for each unique ID.
# We will use groupby() and apply(list) to create a list of labels for each image:

grouped_data = df.groupby('base_ID')['Label'].apply(list).to_dict()

img_list = os.listdir(output_path)

labels=[]

for filename in img_list:
    value = grouped_data[filename[:-4]]
    labels.append(value[-1])

print("Start Cross-Checking if the labels correspond correctly")

# cross-check for random image file #

file_to_check = random.choice(img_list)
index_no = img_list.index(file_to_check)
print(f"{file_to_check} corresponds to the {index_no}th image in the directory")
label = labels[index_no]
print(f"For {file_to_check} the label was assigned: {label}")
print(df[df['base_ID'] == file_to_check.split(".")[0]])
print(f"We should now see a {label} at the ID_any")


# alternative:
# look up all files in the directory img_list in the dictioniary grouped_data
# cut the .png ending of the file which will be they key to search for
# extract the value at index [-1] (= any) and save it in a list
# assert(labels == this_list)

actual_values = []

# Process each file in img_list
for file_name in img_list:
    # Remove the .png extension (or any other extension)
    key = os.path.splitext(file_name)[0]

    # Look up the key in the dictionary and get the last value from the list
    if key in grouped_data:
        value = grouped_data[key][-1]
        actual_values.append(value)

# Assert that labels match this_list
if labels == actual_values:
    print("The labels match this_list!")
else:
    raise AssertionError("The labels do not match this_list!")
