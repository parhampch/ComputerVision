import numpy as np
import cv2
import random
import sys
sys.setrecursionlimit(10 ** 9)


def get_description(n, img, x, y):
    m = n //2
    vector = img[x - m:x + m, y - m:y + m]
    vector = vector.ravel()
    return np.float32(vector)


def get_distance_of_two_points(point1, point2, description_vector1, description_vector2, n):
    d = 0
    for i in range(n ** 2):
        d += (description_vector1[point1][i] - description_vector2[point2][i]) ** 2
    return d ** 0.5


# methods for find components

def dfs(img, index, come_from, max_value, max_point):
    if img[index[0]][index[1]] == 0:
        return ()
    if img[index[0]][index[1]] > max_value:
        max_value = img[index[0]][index[1]]
        max_point = (index[0], index[1])
    img[index[0]][index[1]] = 0
    if index[0] - 1 > -1 and come_from != 'U' and int(img[index[0] - 1][index[1]]) != 0:
        max_value, max_point = dfs(img, (index[0] - 1, index[1]), 'D', max_value, max_point)
    if index[1] + 1 < img.shape[1] and come_from != 'R' and int(img[index[0]][index[1] + 1]) != 0:
        max_value, max_point = dfs(img, (index[0], index[1] + 1), 'L', max_value, max_point)
    if index[0] + 1 < img.shape[0] and come_from != 'D' and int(img[index[0] + 1][index[1]]) != 0:
        max_value, max_point = dfs(img, (index[0] + 1, index[1]), 'U', max_value, max_point)
    if index[1] - 1 > -1 and come_from != 'L' and int(img[index[0]][index[1] - 1]) != 0:
        max_value, max_point = dfs(img, (index[0], index[1] - 1), 'R', max_value, max_point)
    return max_value, max_point


def find_components(img):
    good_points = []
    all_indexes = np.argwhere(img > 0)
    for index in all_indexes:
        good_point = dfs(img, index, 'U', img[index[0]][index[1]], (index[0], index[1]))
        if good_point != ():
            good_points.append(good_point)
    return good_points


# load images

img1 = cv2.imread("im01.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img2 = cv2.imread("im02.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# find derivatives

Ix1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
Ix2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
Iy1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
Iy2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)

# find powers

Ix1Power2 = np.multiply(Ix1, Ix1)
Ix2Power2 = np.multiply(Ix2, Ix2)
Iy1Power2 = np.multiply(Iy1, Iy1)
Iy2Power2 = np.multiply(Iy2, Iy2)
Ix1Iy1 = np.multiply(Ix1, Iy1)
Ix2Iy2 = np.multiply(Ix2, Iy2)

# find gradient and saving them

grad1 = np.sqrt(Ix1Power2 + Iy1Power2)
grad2 = np.sqrt(Ix2Power2 + Iy2Power2)

cv2.imwrite("res01_grad.jpg", grad1)
cv2.imwrite("res02_grad.jpg", grad2)

# enable Gaussian filter

variance = 1.2
pair = (11, 11)
Sx1power2 = cv2.GaussianBlur(Ix1Power2, pair, variance)
Sx2power2 = cv2.GaussianBlur(Ix2Power2, pair, variance)
Sy1power2 = cv2.GaussianBlur(Iy1Power2, pair, variance)
Sy2power2 = cv2.GaussianBlur(Iy2Power2, pair, variance)
Sx1Sy1 = cv2.GaussianBlur(Ix1Iy1, pair, variance)
Sx2Sy2 = cv2.GaussianBlur(Ix2Iy2, pair, variance)

# find determine and trace matrices

determineMatrix1 = np.multiply(Sx1power2, Sy1power2) - np.multiply(Sx1Sy1, Sx1Sy1)
determineMatrix2 = np.multiply(Sx2power2, Sy2power2) - np.multiply(Sx2Sy2, Sx2Sy2)
traceMatrix1 = Sx1power2 + Sy1power2
traceMatrix2 = Sx2power2 + Sy2power2

# find harris function

k = 0.15
R1 = determineMatrix1 - k * np.multiply(traceMatrix1, traceMatrix1)
R2 = determineMatrix2 - k * np.multiply(traceMatrix2, traceMatrix2)
cv2.imwrite("res03_score.jpg", R1)
cv2.imwrite("res04_score.jpg", R2)

# save elements with value more than threshold
threshold = 5000000
R1[R1 < threshold] = 0
R2[R2 < threshold] = 0
cv2.imwrite("res05_thresh.jpg", R1)
cv2.imwrite("res06_thresh.jpg", R2)

# find and handle connected components

n = 35
img1 = cv2.imread("im01.jpg", cv2.IMREAD_COLOR).astype(np.float32)
img2 = cv2.imread("im02.jpg", cv2.IMREAD_COLOR).astype(np.float32)
points1 = find_components(R1)
points2 = find_components(R2)
points1_indexes = []
points2_indexes = []
for i in points1:
    if i[1][0] < n or i[1][0] > img1.shape[0] - n or i[1][1] < n or i[1][1] > img1.shape[1] - n:
        continue
    points1_indexes.append(i[1])
for i in points2:
    if i[1][0] < n or i[1][0] > img1.shape[0] - n or i[1][1] < n or i[1][1] > img1.shape[1] - n:
        continue
    points2_indexes.append(i[1])
for index in points1_indexes:
    img1 = cv2.circle(img1, (index[1], index[0]), 3, (0, 0, 255), 2)
for index in points2_indexes:
    img2 = cv2.circle(img2, (index[1], index[0]), 3, (0, 0, 255), 2)
cv2.imwrite("res07_harris.jpg", img1)
cv2.imwrite("res08_harris.jpg", img2)

# find description vectors

img1 = cv2.imread("im01.jpg", cv2.IMREAD_COLOR).astype(np.float32)
img2 = cv2.imread("im02.jpg", cv2.IMREAD_COLOR).astype(np.float32)
matrix_of_description1 = []
matrix_of_description2 = []
for i in range(len(points1_indexes)):
    matrix_of_description1.append(get_description(n, img1, points1_indexes[i][0], points1_indexes[i][1]))
for i in range(len(points2_indexes)):
    matrix_of_description2.append(get_description(n, img2, points2_indexes[i][0], points2_indexes[i][1]))
matrix_of_description1 = np.float32(np.array(matrix_of_description1))
matrix_of_description2 = np.float32(np.array(matrix_of_description2))

# find corresponding points

corresponding_points_from_img1_to_img2 = {}
corresponding_points_from_img2_to_img1 = {}
final_corresponding_points = {}
threshold = 0.7
for i in range(len(points1_indexes)):
    temp1 = np.float32(matrix_of_description2 - matrix_of_description1[i])
    temp2 = np.float32(np.multiply(temp1, temp1))
    temp3 = np.float32(temp2.transpose().sum(axis=0))
    temp3_prime = np.float32(np.copy(temp3))
    temp4 = temp3.argsort()[:2]
    if temp3_prime[temp4[0]] < (temp3_prime[temp4[1]] * threshold):
        corresponding_points_from_img1_to_img2[points1_indexes[i]] = points2_indexes[temp4[0]]
for i in range(len(points2_indexes)):
    temp1 = matrix_of_description1 - matrix_of_description2[i]
    temp2 = np.multiply(temp1, temp1)
    temp3 = temp2.transpose().sum(axis=0)
    temp3_prime = np.copy(temp3)
    temp4 = temp3.argsort()[:2]
    if temp3_prime[temp4[0]] < (temp3_prime[temp4[1]] * threshold):
        corresponding_points_from_img2_to_img1[points2_indexes[i]] = points1_indexes[temp4[0]]
for point in corresponding_points_from_img2_to_img1.keys():
    temp = corresponding_points_from_img2_to_img1[point]
    if corresponding_points_from_img1_to_img2.keys().__contains__(temp) and point == corresponding_points_from_img1_to_img2[temp]:
        final_corresponding_points[point] = temp

# remove duplicate values

temp_dic = {}
for key in final_corresponding_points.keys():
    if temp_dic.keys().__contains__(key):
        temp_dic[key] += 1
    else:
        temp_dic[key] = 1
for key in temp_dic:
    if temp_dic[key] > 1:
        final_corresponding_points.pop(key)

# show corresponding points in each picture

org_img1 = cv2.imread("im01.jpg", cv2.IMREAD_COLOR).astype(np.float32)
org_img2 = cv2.imread("im02.jpg", cv2.IMREAD_COLOR).astype(np.float32)
for point in final_corresponding_points.keys():
    org_img1 = cv2.circle(org_img2, (point[1], point[0]), 3, (0, 0, 255), 2)
    org_img2 = cv2.circle(org_img1, (final_corresponding_points[point][1], final_corresponding_points[point][0]), 3, (0, 0, 255), 2)
cv2.imwrite("res09_corres.jpg", org_img1)
cv2.imwrite("res10_corres.jpg", org_img2)

# concat images

org_img1 = cv2.imread("im01.jpg", cv2.IMREAD_COLOR).astype(np.float32)
org_img2 = cv2.imread("im02.jpg", cv2.IMREAD_COLOR).astype(np.float32)
fin_img = cv2.hconcat([org_img1, org_img2])
for point in final_corresponding_points.keys():
    p = (point[1] + org_img1.shape[1], point[0])
    fin_img = cv2.line(fin_img, (final_corresponding_points[point][1], final_corresponding_points[point][0]), p, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
cv2.imwrite("res11.jpg", fin_img)