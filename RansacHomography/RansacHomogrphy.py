import random
import numpy as np
import cv2


def get_matrix_for_a_pair(src, dst):
    result = []
    result.append([0, 0, 0, -src[0], -src[1], -1, dst[1] * src[0], dst[1] * src[1], dst[1]])
    result.append([src[0], src[1], 1, 0, 0, 0, -dst[0] * src[0], -dst[0] * src[1], -dst[0]])
    return result


def get_random_indexes(src_point):
    result = []
    l = len(src_point) - 1
    result.append(random.randint(0, l))
    result.append(random.randint(0, l))
    result.append(random.randint(0, l))
    result.append(random.randint(0, l))
    return result


def ransac(src_points, dst_points):
    threshold = 5
    best_sample = []
    max_number = 0
    number_of_repitation = 100000
    matrix_of_src = []
    matrix_of_dst = []
    for point in src_points:
        point.append(1)
        matrix_of_src.append(point)
    for point in dst_points:
        point.append(1)
        matrix_of_dst.append(point)
    matrix_of_src = np.array(matrix_of_src)
    matrix_of_dst = np.array(matrix_of_dst)
    for i in range(number_of_repitation):
        indexes = get_random_indexes(src_points)
        A = []
        A.append(get_matrix_for_a_pair(src_points[indexes[0]], dst_points[indexes[0]]))
        A.append(get_matrix_for_a_pair(src_points[indexes[1]], dst_points[indexes[1]]))
        A.append(get_matrix_for_a_pair(src_points[indexes[2]], dst_points[indexes[2]]))
        A.append(get_matrix_for_a_pair(src_points[indexes[3]], dst_points[indexes[3]]))
        A = np.array(A).reshape((-1, 9))
        u, sigma, v_t = np.linalg.svd(A)
        h = v_t[-1]
        h = h / h[8]
        h = h.reshape((3, 3))
        temp = np.dot(h, matrix_of_src.transpose())
        if not(temp[-1].all()):
            continue
        temp = np.divide(temp, temp[-1])
        temp = temp.transpose() - matrix_of_dst
        temp = np.multiply(temp, temp)
        temp = np.sum(temp, axis=1)
        temp = np.sqrt(temp)
        temp = np.argwhere(temp < threshold).reshape((-1))
        s = temp.size
        if s > max_number:
            max_number = s
            best_sample = np.copy(temp)
    A = []
    for index in best_sample:
        A.append(get_matrix_for_a_pair(src_points[index], dst_points[index]))
    A = np.array(A).reshape((-1, 9))
    u, sigma, v_t = np.linalg.svd(A)
    result = v_t[-1]
    result = result / result[8]
    return result.reshape((3, 3))


# load images

img1 = cv2.imread("im03.jpg")
img2 = cv2.imread("im04.jpg")

# find interest point

sift = cv2.SIFT_create()
points1, description1 = sift.detectAndCompute(img1, None)
points2, description2 = sift.detectAndCompute(img2, None)
points1_indexes = []
points2_indexes = []
for point in points1:
    points1_indexes.append(point.pt)
for point in points2:
    points2_indexes.append(point.pt)

# find corresponding points

brute_force = cv2.BFMatcher()
corresponding_pairs = brute_force.knnMatch(description1, description2, k=2)
good_corresponding_points = []
for i, j in corresponding_pairs:
    if i.distance < (0.7 * j.distance):
        good_corresponding_points.append(i)
corresponding_points_from_picture1 = []
corresponding_points_from_picture2 = []
for point in good_corresponding_points:
    corresponding_points_from_picture1.append(list(points1[point.queryIdx].pt))
    corresponding_points_from_picture2.append(list(points2[point.trainIdx].pt))

# compute homography matrix

H = ransac(corresponding_points_from_picture2, corresponding_points_from_picture1)
print(H)
# transpose matrix and create result
transpose_matrix = np.array([[1, 0, 3000], [0, 1, 3000], [0, 0, 1]])
H = np.dot(transpose_matrix, H)
img1 = cv2.imread("im03.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("im04.jpg", cv2.IMREAD_COLOR)
dst = cv2.warpPerspective(img2, H, (7 * img1.shape[1], 3 * img1.shape[0]))
cv2.imwrite("res20.jpg", dst)



