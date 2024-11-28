import os
import glob

# Specify the directory path
directory = "/home/sebastian/code/ipl1988/raw_data/stage_2_test"

# Specify the file pattern (e.g., "IDENTIFIER")
file_pattern = os.path.join(directory, "*Identifier")  # Match files ending with "IDENTIFIER"

# Find and delete the matching files
for file_path in glob.glob(file_pattern):
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
