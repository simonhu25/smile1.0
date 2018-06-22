'''
Version: 1.0
Date: 06.21.2018
By: Jun Hao Hu @ University of California San Diego
All rights reserved.
'''

from time import time
import logging
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import math

from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import fetch_olivetti_faces

from keras.utils import to_categorical

from scipy.stats import sem

# print details of the program
print(__doc__)

# display progress logs on sdtout
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

###############################################################################

'''
Download the Olivetti Faces dataset.
The dataset contains 400 images each of size 64 x 64 pixels.
The dataset is already normalized to the 0-1 range.
faces.images is of shape (400,64,64).
faces.data is the same as faces.images but is of shape (400,4096).
faces.target is of shape (400,) and contains the labels denoting whether the person is smiling or not.
'''
faces = fetch_olivetti_faces()

###############################################################################

'''
Build the training and testing sets.
'''
X_train, X_test, Y_train, Y_test = train_test_split(faces.data,faces.target,test_size=0.2,random_state=5)

###############################################################################

'''
Set up the SVM classifier and fit it to the training data.
The best hyperparameters for this setup are: linear, gamma = 0.0001, C = 1000.0.
'''
param_grid = {'C':[1e3,5e3,1e5,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
t0 = time()
print('\nTraining the SVM classifier\n')
clf = SVC(kernel='linear',C=1000.0,gamma=0.0001)
clf = clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))
