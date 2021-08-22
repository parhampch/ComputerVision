import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn import metrics, svm
import matplotlib.pyplot as plt
import pickle


win_size = (128, 128)
block_size = (16, 16)
cell_size = (8, 8)
block_stride = (8, 8)

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)
train_data_desc = []
train_data_category = []

categories = os.listdir("Data/Train")
for category in tqdm(categories):
    path = "Data/Train/" + category
    images = glob.glob(path + "/*.jpg")
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, win_size)
        desc = hog.compute(img)
        train_data_desc.append(desc.reshape(-1))
        train_data_category.append(category)

train_data_desc = np.array(train_data_desc)
train_data_category = np.array(train_data_category)

svm_ins = svm.SVC(kernel='rbf', probability=True)
svm_ins.fit(train_data_desc, train_data_category)
pickle.dump(svm_ins, open('svm_model.sav', 'wb'))

# svm_ins = pickle.load(open('svm_model.sav', 'rb'))

validation_data_desc = []
validation_data_category = []

categories = os.listdir("Data/Validation")
for category in tqdm(categories):
    path = "Data/Validation/" + category
    images = glob.glob(path + "/*.jpg")
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, win_size)
        desc = hog.compute(img)
        validation_data_desc.append(desc.reshape(-1))
        validation_data_category.append(category)
validation_data_desc = np.array(validation_data_desc)
validation_data_category = np.array(validation_data_category)

result = svm_ins.predict(validation_data_desc)
true = 0
for i in tqdm(range(0, len(validation_data_category))):
    if validation_data_category[i] == result[i]:
        true += 1
percent = (true / len(validation_data_category)) * 100
print("Validation precision is : ", percent)

test_data_desc = []
test_data_category = []

categories = os.listdir("Data/Test")
for category in tqdm(categories):
    path = "Data/Test/" + category
    images = glob.glob(path + "/*.jpg")
    for image in images:
        img = cv2.imread(image)
        img = cv2.resize(img, win_size)
        desc = hog.compute(img)
        test_data_desc.append(desc.reshape(-1))
        test_data_category.append(category)
test_data_desc = np.array(test_data_desc)
test_data_category = np.array(test_data_category)

result = svm_ins.predict(test_data_desc)
true = 0
for i in tqdm(range(0, len(test_data_category))):
    if test_data_category[i] == result[i]:
        true += 1
percent = (true / len(test_data_category)) * 100
print("Test precision is : ", percent)

metrics.plot_roc_curve(svm_ins, test_data_desc, test_data_category)
plt.savefig("res1.jpg")

metrics.plot_precision_recall_curve(svm_ins, test_data_desc, test_data_category)
plt.savefig("res2.jpg")
