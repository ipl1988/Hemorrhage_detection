from basic_preprocessing import transform_in_hu, normalizer, process_images


## Draw 100 random samples from #

# Define the img path!!! -> img path = folder containing your DICOM files
img_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_test"

## For later modelling, we will need a directory 2 levels down
output_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_test_proc/images"

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)



process_images(100)
