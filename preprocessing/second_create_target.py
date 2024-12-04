import os
from dotenv import load_dotenv
import pandas as pd

# Load variables from the .env file
load_dotenv()
output_path = os.getenv("output_path")

# ## Read in csv containing target labels of training data

df = pd.read_csv("../training_data_labels.csv")
dictionary = df.set_index('Key')['Value'].to_dict()
img_list = os.listdir(output_path)

labels=[]
actual_values = []

for file_name in img_list:
    # Remove the .png extension (or any other extension)
    key = os.path.splitext(file_name)[0]

    if key in dictionary:
        # Get the last label from the list for the current file
        label = int(dictionary[key][-2])
        true_value = int(df[df['Key'] == key]['Value'].iloc[0][-2])

        # Append the label to both lists
        labels.append(label)
        actual_values.append(true_value)
    else:
        # Handle the case where the key is not found in the grouped data
        print(f"Warning: {key} not found in grouped_data")

##### Extract the Image ID

# Assuming the ID column contains both the image ID (e.g., ID_12cadc6af) and the type of hemorrhage (e.g., epidural, intraparenchymal),
# we need to extract just the base ID. The base ID is the part before the first underscore.
# This can be done using the str.extract() method to grab everything up to the first underscore

#df['base_ID'] = df['ID'].str.extract(r'^(ID_[^_]+)')

#### Group the Labels by the Base Image ID

# Now, we can group the DataFrame by this base ID and collect all the corresponding Label values for each unique ID.
# We will use groupby() and apply(list) to create a list of labels for each image:

# grouped_data = df.groupby('base_ID')['Label'].apply(list).to_dict()

# img_list = os.listdir(output_path)

# labels=[]
# actual_values = []

# for file_name in img_list:
#     # Remove the .png extension (or any other extension)
#     key = os.path.splitext(file_name)[0]

#     if key in grouped_data:
#         # Get the last label from the list for the current file
#         label = grouped_data[key][-1]
#         true_value = df[df['ID'] == key + "_any"]['Label'].iloc[0]

#         # Append the label to both lists
#         labels.append(label)
#         actual_values.append(true_value)
#     else:
#         # Handle the case where the key is not found in the grouped data
#         print(f"Warning: {key} not found in grouped_data")

print("Start Cross-Checking if the labels correspond correctly")

# Assert that labels match this_list
if labels == actual_values:
    print("The labels match this_list!")
else:
    raise AssertionError("The labels do not match this_list!")



## Optional: Export Labels ##

# Convert the 'labels' list into a DataFrame

#labels_df = pd.DataFrame(labels, columns=['Label'])

# Specify the path where you want to save the CSV and save it
# output_csv_path = "../testdatalabels.csv"
# labels_df.to_csv(output_csv_path, index=False)
# print(f"Labels have been successfully exported to {output_csv_path}")
