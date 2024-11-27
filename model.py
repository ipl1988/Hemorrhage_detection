import os
# from create_target import labels
# from basic_preprocessing import output_path

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential

import numpy as np

#processed_image_list = os.listdir(output_path)
# for label, image in zip(labels, processed_image_list):
# print(label,image)


### Move all files in output_path to an extra directory, otherwise tf can't read them
# delete 28th 11
# img_dir_path = output_path[:-7]

labels = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
img_dir_path = "/home/sebastian/code/ipl1988/raw_data/stage_2_train/images_stage_2_train_png"

print(labels)
print(len(labels))
print(os.listdir(output_path))

dataset = image_dataset_from_directory(
    directory = img_dir_path,
    labels=labels,
    label_mode='int',
    class_names=None,
    color_mode='grayscale',
    batch_size=32,
    image_size=(150,150),
    shuffle=True,
    seed=None,
    validation_split=None,
)

for images, labels in dataset.take(1):  # take(1) gives you just the first batch
    print(images.shape)  # The shape of the images (e.g., (batch_size, 256, 256, 1))
    print(labels.shape)  # The shape of the labels (e.g., (batch_size, 1) for binary)
    print(labels)        # The actual labels for this batch


model = Sequential()

model.add(layers.Input((150, 150, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", padding="same"))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.fit(dataset)