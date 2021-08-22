import cv2
import numpy as np
from tqdm import tqdm
import pickle
from sklearn import metrics, svm

win_size = (128, 128)
block_size = (16, 16)
cell_size = (8, 8)
block_stride = (8, 8)


def find_face(img, scale):
    global win_size
    global hog
    global svm_ins
    img_shape = img.shape
    good_coordinates = []
    for i in tqdm(range(0, img_shape[0] - win_size[0], 10)):
        for j in range(0, img_shape[1] - win_size[1], 10):
            sub_img = img[i:i + win_size[0], j:j + win_size[1]]
            desc = hog.compute(sub_img)
            res = svm_ins.predict_proba(np.array([desc.reshape(-1)]))
            if res[0][1] >= 0.9:
                good_coordinates.append((j, i, scale, res[0][1]))
    return good_coordinates


def FaceDetector(img):
    scale = 1
    multiplier = 1
    final_coordinates = []
    for i in range(6):
        img_shape = img.shape
        new_size = (int(scale * img_shape[1]), int(scale * img_shape[0]))
        new_img = cv2.resize(img, new_size)
        final_coordinates.append(find_face(new_img, scale))
        scale -= 0.1
    temp = final_coordinates
    final1 = []
    for set_of_coor in temp:
        for coor in set_of_coor:
            scale = int(1 / coor[2])
            final1.append((scale * coor[0], scale * coor[1], coor[2], coor[3]))
    final2 = []
    for i in range(len(final1)):
        coor1 = final1[i]
        goof_for_add = True
        for j in range(len(final2)):
            coor2 = final2[j]
            a = 0
            b = 0
            if coor1[0] < coor2[0]:
                if coor2[0] >= coor1[0] + int(1 / coor1[2]) * win_size[0]:
                    continue
                a = coor1[0] + int(1 / coor1[2]) * win_size[0] - coor2[0]
                if coor1[1] < coor2[1]:
                    b = int(1 / coor1[2]) * win_size[1] + coor1[1] - coor2[1]
                else:
                    b = int(1 / coor2[2]) * win_size[1] + coor2[1] - coor1[1]
            else:
                if coor1[0] >= coor2[0] + int(1 / coor2[2]) * win_size[0]:
                    continue
                a = coor2[0] + int(1 / coor2[2]) * win_size[0] - coor1[0]
                if coor2[1] < coor1[1]:
                    b = int(1 / coor2[2]) * win_size[1] + coor2[1] - coor1[1]
                else:
                    b = int(1 / coor1[2]) * win_size[1] + coor1[1] - coor2[1]
            subscription = a * b
            S = int(1 / coor1[2]) * win_size[0] * (int(1 / coor1[2]) * win_size[1])
            S += int(1 / coor2[2]) * win_size[0] * (int(1 / coor2[2]) * win_size[1])
            union = S - subscription
            if (subscription / union) > 0.075:
                goof_for_add = False
                break
        if goof_for_add:
            final2.append(coor1)
    for coor in final2:
        scale = int(1 / coor[2])
        img = cv2.rectangle(img, (coor[0], coor[1]), (coor[0] + scale * win_size[0], coor[1] + scale * win_size[1]), (0, 255, 0), 3)
    return img

svm_ins = pickle.load(open('svm_model.sav', 'rb'))
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 9)

img = cv2.imread("Melli.jpg")
new_img = FaceDetector(img)
cv2.imwrite("res4.jpg", new_img)

img = cv2.imread("Persepolis.jpg")
new_img = FaceDetector(img)
cv2.imwrite("res5.jpg", new_img)

img = cv2.imread("Esteghlal.jpg")
new_img = FaceDetector(img)
cv2.imwrite("res6.jpg", new_img)