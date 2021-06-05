import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

histo_length = 75

categories = os.listdir("Data/Train")
sift = cv2.SIFT_create()
description_vectors_train = []
for category in tqdm(categories):
    path = "Data/Train/" + category
    files = glob.glob(path + "/*.jpg")
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        points, description = sift.detectAndCompute(img, None)
        description_vectors_train.append(description)
description_vectors_train = np.vstack(description_vectors_train)
kmeans = KMeans(n_clusters=histo_length, random_state=0, n_init=1, max_iter=300).fit(description_vectors_train)
centers = kmeans.cluster_centers_

X_train = []
y_train = []
for category in tqdm(categories):
    path = "Data/Train/" + category
    files = glob.glob(path + "/*.jpg")
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        points, description = sift.detectAndCompute(img, None)
        histo = np.zeros((1, histo_length))[0]
        for i in range(len(description)):
            temp = centers - description[i]
            temp = np.multiply(temp, temp)
            temp = np.sum(temp, axis=1)
            histo[np.argmin(temp)] += 1
        X_train.append(histo)
        y_train.append(category)
X_train = np.array(X_train)
y_train = np.array(y_train)


X_test = []
y_test = []
categories = os.listdir("Data/Test")
for category in tqdm(categories):
    path = "Data/Test/" + category
    files = glob.glob(path + "/*.jpg")
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        points, description = sift.detectAndCompute(img, None)
        histo = np.zeros((1, histo_length))[0]
        for i in range(len(description)):
            temp = centers - description[i]
            temp = np.multiply(temp, temp)
            temp = np.sum(temp, axis=1)
            histo[np.argmin(temp)] += 1
        X_test.append(histo)
        y_test.append(category)
X_test = np.array(X_test)
y_test = np.array(y_test)

svm_ins = svm.SVC()
svm_ins.fit(X_train, y_train)

result = svm_ins.predict(X_test)
true = 0
for i in tqdm(range(0, len(y_test))):
    if y_test[i] == result[i]:
        true += 1
percent = (true / len(y_test)) * 100
print(percent)


