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

'''
Download the images.
These images will be uniformly resized to 100 x 100 (height, width) and stored in a numpy array.
They will then be fed into a neural network
'''
h = 100
w = 100
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
TODO: Continue with processing the images until you can set up the neural network. SVM might be a better approach to this problem
'''
