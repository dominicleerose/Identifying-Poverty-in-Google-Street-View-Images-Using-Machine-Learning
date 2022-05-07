# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 23:09:59 2022

@author: 40544
"""
## Convolutional neural network (CNN) to classify images

## Explanation of how a CNN works: https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d#:~:text=Convolutional%20Neural%20Networks%20(CNNs)%20is,used%20for%20image%20classification%20problem.

# Load packages

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Load data

DATADIR = "C:/Users/domin/Documents/LSE/Capstone/Testing ML Methods/Test Disso Code/basedata/training"
CATEGORIES = ["high-income", "medium-income", "low-income"]

# Preview an image

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # Path to income directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array, cmap = "gray")
        plt.show()
        break
    break

print(img_array)
print(img_array.shape) # 5333 pixel height x 8000 pixel width

# Normalise image size

IMG_SIZE = 224

# Resize the image
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = "gray")
plt.show()

# Create training data

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # Path to income directory
        class_num = CATEGORIES.index(category) # Index the categories
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data)) # Should be balanced given three categories

import random

# Randomise the data

random.shuffle(training_data)

# Show example of shuffled data

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# -1 represents the amount of features in the dataset; 3 specifies that we are working with coloured images. (IMG_SIZE, IMG_SIZE) represents the shape of the data.

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

import pickle

# Save data to external files

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# Read data 

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

## Building the CNN 

# Load packages

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Convert data to np.array

X = np.array(pickle.load(open("X.pickle", "rb")))
y = np.array(pickle.load(open("y.pickle", "rb")))

# Colour scale ranges from 0-255. Therefore, divide X by 255 to normalise the training images.

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:])) # Convolutional layer
model.add(Activation("relu")) # Activation layer
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3))) # Second convolutional layer 
model.add(Activation("relu")) # Activation layer
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten()) # Flatten to one-dimensional data 
model.add(Dense(64))
model.add(Activation("relu")) # Activation layer

model.add(Dense(1))
model.add(Activation("softmax"))

# Use the categorical_crossentropy loss function, the Adam optimizer and accuracy to measure model performance 

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

# X represents my training_images and y represents my training labels

# Can adjust batch size and epochs. Validation split = 10%.

history = model.fit(X, y, batch_size = 32, epochs = 5, validation_split = 0.1)

# Model evaluation plot creation

plt.plot(history.history["accuracy"], label = "accuracy")
plt.plot(history.history["val_accuracy"], label = "val_accuracy")
plt.title("CNN Model Accuracy Values")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.1, 0.4])
plt.legend(loc = "lower right")

# Evaluate model

test_loss, test_acc = model.evaluate(X, y, verbose = 2)

print(test_acc)