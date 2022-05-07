# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:34:29 2022

@author: 40544
"""
## Support vector machine (SVM) to classify images

# Load packages

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set directory of training images

dir = "C:/Users/domin/Documents/LSE/Capstone/Testing ML Methods/Test Disso Code/basedata/training"

# List the classes

categories = ["high-income", "medium-income", "low-income"]

# Load data

data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        income_img = cv2.imread(imgpath, 0)
        try:
            income_img = cv2.resize(income_img, (224, 224))
            image = np.array(income_img).flatten()
            
            data.append([image, label])
        except Exception as e:
            pass

print(len(data)) # Total number of images

# Save data
        
pick_in = open("data1.pickle", "wb")
pickle.dump(data, pick_in)
pick_in.close()

# Load data

pick_in = open("data1.pickle", "rb")
data = pickle.load(pick_in)
pick_in.close()

# Randomise the data

random.shuffle(data)
features = []
labels = []

# Combine data features and labels

for feature, label in data:
    features.append(feature)
    labels.append(label)

# Create training and test sets. Test set = 25%.

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25)

# Create support vector machine model

model = SVC(C = 1, kernel = "poly", gamma = "auto")
model.fit(xtrain, ytrain) # Fit the model

# Save data

pick = open("model.sav", "wb")
pickle.dump(model, pick)
pick.close()

# Load data

pick = open("model.sav", "rb")
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest) # Apply model on test set

accuracy = model.score(xtest, ytest) # Generate accuracy score

# Specify classes (income categories)

categories = ["high-income", "medium-income", "low-income"]

print("Accuracy is: ", accuracy)

# Predict an image as low-, medium, or high-income

print("Prediction is: ", categories[prediction[0]])

# Preview image

income_picture = xtest[0].reshape(224, 224)
plt.imshow(income_picture)
plt.show()