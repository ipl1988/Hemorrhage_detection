import pydicom as dicom
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd

## Specify the path to your target CSV file
data_path = "/home/sebastian/code/ipl1988/raw_data"
# Read the target CSV file
df = pd.read_csv(os.path.join(data_path, "stage_2_train.csv"))

##### Extract the Image ID

# Assuming the ID column contains both the image ID (e.g., ID_12cadc6af) and the type of hemorrhage (e.g., epidural, intraparenchymal),
# we need to extract just the base ID. The base ID is the part before the first underscore.
# This can be done using the str.extract() method to grab everything up to the first underscore

df['base_ID'] = df['ID'].str.extract(r'^(ID_[^_]+)')

#### Group the Labels by the Base Image ID

# Now, we can group the DataFrame by this base ID and collect all the corresponding Label values for each unique ID.
# We will use groupby() and apply(list) to create a list of labels for each image:

grouped_data = df.groupby('base_ID')['Label'].apply(list).to_dict()

#### IMPORT OUPUT PATH FIRST!!!!!
raw_img_list = os.listdir("/home/sebastian/code/ipl1988/raw_data/stage_2_train/images_stage_2_train_png") ## from preprocessing!

labels=[]

for filename in raw_img_list:
    value = grouped_data[filename[:-4]]
    labels.append(value[-1])

print(raw_img_list, labels)

print("Start Cross-Checking if the labels correspond correctly")

# For File ID_85a3177f2 #

file_to_check = "ID_85a3177f2"
index_no = raw_img_list.index(file_to_check + '.png')
print(f"{file_to_check} corresponds to the {index_no}th image in the directory")
label = labels[index_no]
print(f"For {file_to_check} the label was assigned: {label}")
print(df[df['base_ID'] == file_to_check])
print(f"We should now see a {label} at the ID_any")
