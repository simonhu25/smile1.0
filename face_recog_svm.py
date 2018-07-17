'''
Version: 1.0
Date: 06.20.2018
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
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
labels = np.load('truth_labels.npy').ravel()
#labels = to_categorical(labels,num_classes=2)

'''
Download the images.
These images will be uniformly resized to 100 x 100 (height, width) and stored in a numpy array.
'''
h = 64
w = 64
prod = h*w

image_names = glob.glob('./images/*')
n_images = len(image_names)

images = np.array([np.array([(cv2.resize(cv2.imread(file,0),(h,w),cv2.INTER_LANCZOS4)/255.0) for file in image_names])])[0]
images *= 255.0/images.max()

'''
Split the datasets (images and labels) into training and testing sets.
The following split will be used: 80% of the dataset will be used as training data. 20% of the dataset will be used as testing data
'''
random_seed = 2
X_train,X_test,Y_train,Y_test = train_test_split(images,labels,test_size=0.2,random_state=random_seed)

'''
Flatten the images so that they are of the form n_images x (h * w)
'''
n_train = X_train.shape[0]
n_test = X_test.shape[0]
X_train = X_train.flatten().reshape(n_train,prod)
X_test = X_test.flatten().reshape(n_test,prod)

###############################################################################

'''
Compute a PCA on the images.
The optimal number of principal components will be determined through trial-and-error.
First iteration: 200 components
'''
n_components = 400

print("\nExtracting the top %d eigenfaces from %d faces." % (n_components,X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components,whiten=True,svd_solver='randomized',random_state=random_seed).fit(X_train)
print("\nExtraction completed. Time elapsed: %0.4fs." % (time() - t0))

eigenfaces = pca.components_.reshape((n_components,h,w))

print("\nProjecting onto the eigenbasis.")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("\nProjection completed in: %0.4fs" % (time() - t0))

###############################################################################

'''
Normalize the PCA images, since we are going to be feeding them into the SVM.
'''
#X_train_pca

##############################################################################

'''
Train a SVM to classify the images into "smile" and "not smile".
Perform a Grid-Search to find the best hyper-parameters.
'''

param_grid = {'C':[1e3,5e3,1e5,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1]}
t0 = time()
print('\nTraining the SVM classifier.\n')
clf = GridSearchCV(SVC(kernel='linear',class_weight=None),param_grid,verbose=0)
clf = clf.fit(X_train_pca,Y_train)
print("Fitting done in: %0.4fs" % (time() - t0))
print("The parameters are: ")
print(clf.best_estimator_)
