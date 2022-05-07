# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 20:54:56 2022

@author: 40544
"""
## Transfer learning to classify images

# Load packages

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

IMAGE_SIZE = [224, 224] # Standardise image size

# Set paths for training and testing data. Local paths are used here.

train_path = "C:/Users/domin/Documents/LSE/Capstone/Testing ML Methods/Test Disso Code/basedata/training"
test_path = "C:/Users/domin/Documents/LSE/Capstone/Testing ML Methods/Test Disso Code/basedata/testing"

from PIL import Image 
import os 
from IPython.display import display
from IPython.display import Image as _Imgdis

# Display low-income images as an example
  
folder = train_path + "/low-income"

# Get list of all directories inside the low-income folder as an example to see how many exist

only_low_income_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print("Working with {0} images".format(len(only_low_income_files)))
print("Image examples: ") # Number of low-income images

for i in range(5):
    print(only_low_income_files[i])
    display(_Imgdis(filename = folder + "/" + only_low_income_files[i], width = 240, height = 240))

# Use VGG16 convolutional neural network and the weights from the ImageNet classification challenge

vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = "imagenet", include_top = False)

vgg.input

for layer in vgg.layers:
    layer.trainable = False # Do not re-train layers, just use parameters that the model has learned

# Number of folders/classes (3 - i.e. high-income, medium-income and low-income)
    
folders = glob("C:/Users/domin/Documents/LSE/Capstone/Testing ML Methods/Test Disso Code/basedata/training/*")
print(len(folders))

# Create and summarise. Using transfer learning, we have 14,789,955 parameters. Reduces computational complexity.

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = "softmax")(x)
model = Model(inputs = vgg.input, outputs = prediction)
model.summary()

from keras import optimizers

# Use the categorical_crossentropy loss function, the Adam optimizer and accuracy to measure model performance 

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

# Pre-process the images in training and testing sets

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest")

test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = "nearest")

# Can change batch_size here - i.e. the number of batches that the model will train on at a time

train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 4,
                                                 class_mode = "categorical")

# test_set is half of train_set

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 4,
                                            class_mode = "categorical")

# Save the best model using ModelCheckpoint

from datetime import datetime
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath = "mymodel.h5", 
                               verbose = 2, save_best_only = True)

callbacks = [checkpoint]

start = datetime.now() # How long training process takes

model_history = model.fit_generator(
  train_set,
  validation_data = test_set,
  epochs = 5,
  steps_per_epoch = 2,
  validation_steps = 32,
    callbacks = callbacks, verbose = 2)

duration = datetime.now() - start
print("Training completed in time: ", duration)

# Plot training and validation loss values

plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.title("CNN Model Accuracy Values With Transfer Learning")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc = "lower right")
plt.show()