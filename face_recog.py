'''
Version: 1.0
Date: 06.20.2018
By: Jun Hao Hu @ University of California San Diego
All rights reserved.
'''

from time import time
import logging

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

from keras.utils import to_categorical

# print details of the program
print(__doc__)

# display progress logs on sdtout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

###############################################################################

'''
Download the labels.
These categorical features will be encoded using one-hot encoding.
'''
labels = np.load('truth_labels.npy')
labels = to_categorical(labels,num_classes=2)

'''
Download the images.
These images will be uniformly resized to 100 x 100 (height, width) and stored in a numpy array.
'''
h = 100
w = 100
image_names = glob.glob('./images/*')
n_images = len(image_names)
images = np.array([np.array([cv2.resize(cv2.imread(file),(h,w),cv2.INTER_LANCZOS4) for file in image_names])])[0]

'''
Split the datasets (images and labels) into training and testing sets.
The following split will be used: 80% of the dataset will be used as training data. 20% of the dataset will be used as testing data
'''
split_idx = math.floor(0.80*n_images)
train_data = images[0:split_idx]
test_data = images[split_idx:n_images+1]
train_label = labels[0:split_idx]
test_label = labels[split_idx:n_images+1]

'''
Split the training data into training and validation datasets.
The following split will be used: 80% of the dataset will be used as training data. 20% of the dataset will be used as validation data
Shuffle the data using the random seed.
'''
random_seed = 5
X_train,X_val,Y_train,Y_val = train_test_split(train_data,train_label,test_size=0.1,random_state=random_seed)
