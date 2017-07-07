#!/usr/bin/python3

import cv2
import math
import pickle
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC

PADDING = 10
TARGET_SIZE = 50

with open('glacierdigits2.glacierdigits2.exp0.box', 'r') as f:
    lines = f.read().strip().split('\n')
    data = [ list(map(int, line.split(' '))) for line in lines ]

im = cv2.imread('glacierdigits2.glacierdigits2.exp0.tif', cv2.IMREAD_GRAYSCALE)
h, w = im.shape

data_features = []
data_labels = []
for d in data:
    num, x1, y2, x2, y1, _ = d
    y1 = im.shape[0] - y1
    y2 = im.shape[0] - y2

    char = im[y1:y2, x1:x2]
    char = cv2.resize(char, (TARGET_SIZE, TARGET_SIZE), cv2.INTER_CUBIC)
    
    #cv2.imshow('img', char)
    #cv2.waitKey()

    data_features.append( hog(char) )
    data_labels.append( num )

data_features = np.array(data_features, dtype=np.float64)
data_labels = np.array(data_labels, dtype=np.int32)

#print(data_features.shape)
#print(data_labels.shape)

model = LinearSVC()
model.fit(data_features, data_labels)
error1 = model.score(data_features, data_labels)

with open('model.pkl', 'wb') as f:
    f.write( pickle.dumps(model) )

with open('model.pkl', 'rb') as f:
    model2 = pickle.loads(f.read())
    error2 = model2.score(data_features, data_labels)
    assert(error1 == error2)
    print('Training error', error1)
