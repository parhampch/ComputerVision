import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

images_size = (14, 14)

X_train = []
y_train = []
categories = os.listdir("Data/Train")
for category in tqdm(categories):
    path = "Data/Train/" + category
    files = glob.glob(path + "/*.jpg")
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, images_size)
        vec = img.reshape(-1)
        X_train.append(vec)
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
        img = cv2.resize(img, images_size)
        vec = img.reshape(-1)
        X_test.append(vec)
        y_test.append(category)
X_test = np.array(X_test)
y_test = np.array(y_test)

knn = KNeighborsClassifier(n_neighbors=1, p=1)
knn.fit(X_train, y_train)

result = knn.predict(X_test)
true = 0
for i in tqdm(range(0, len(y_test))):
    if y_test[i] == result[i]:
        true += 1
percent = (true / len(y_test)) * 100
print(percent)