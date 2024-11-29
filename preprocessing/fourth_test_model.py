from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd

# Load the saved model (point to the directory)
model = load_model('./model_V3_windowed')

img_test_dir_path = "/home/sebastian/code/ipl1988/test_data_dont_touch"
labels = pd.read_csv("/home/sebastian/code/ipl1988/test_data_labels.csv")['Label'].values

test_dataset = image_dataset_from_directory(
    directory=img_test_dir_path,  # Path to test images
    labels=list(labels),  # Labels for the test data
    label_mode='int',  # 'int' for integer labels
    class_names=None,  # Can be left as None if labels are already provided
    color_mode='grayscale',
    batch_size=32,
    image_size=(150, 150),
    shuffle=False  # Set to False to preserve the order
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
