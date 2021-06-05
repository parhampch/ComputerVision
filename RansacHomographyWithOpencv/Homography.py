import numpy as np
import cv2

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
img1 = cv2.drawKeypoints(img1, points1, img1, (0, 255, 0))
img2 = cv2.drawKeypoints(img2, points2, img2, (0, 255, 0))
img2 = cv2.copyMakeBorder(
    img2,
    top=504,
    bottom=504,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT
)
result1 = cv2.hconcat([img1, img2])
cv2.imwrite("res13_corners.jpg", result1)

# find corresponding points

brute_force = cv2.BFMatcher()
corresponding_pairs = brute_force.knnMatch(description1, description2, k=2)
good_corresponding_points = []
for i, j in corresponding_pairs:
    if i.distance < (0.75 * j.distance):
        good_corresponding_points.append(i)
corresponding_points_from_picture1 = []
corresponding_points_from_picture2 = []
for point in good_corresponding_points:
    corresponding_points_from_picture1.append(points1[point.queryIdx].pt)
    corresponding_points_from_picture2.append(points2[point.trainIdx].pt)
img1 = cv2.imread("im03.jpg")
img2 = cv2.imread("im04.jpg")
for index in points1_indexes:
    if corresponding_points_from_picture1.__contains__(index):
        img1 = cv2.circle(img1, (int(index[0]), int(index[1])), 3, (255, 0, 0), 2)
    else:
        img1 = cv2.circle(img1, (int(index[0]), int(index[1])), 3, (0, 255, 0), 2)
for index in points2_indexes:
    if corresponding_points_from_picture2.__contains__(index):
        img2 = cv2.circle(img2, (int(index[0]), int(index[1])), 3, (255, 0, 0), 2)
    else:
        img2 = cv2.circle(img2, (int(index[0]), int(index[1])), 3, (0, 255, 0), 2)
img2 = cv2.copyMakeBorder(
    img2,
    top=504,
    bottom=504,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT
)
result2 = cv2.hconcat([img1, img2])
cv2.imwrite("res14_correspondences.jpg", result2)

# match corresponding points with line

img1 = cv2.imread("im03.jpg")
img2 = cv2.imread("im04.jpg")
draw_params = dict(matchColor=(255, 0, 0),
                   singlePointColor=None,
                   matchesMask=None,
                   flags=2)
result3 = cv2.drawMatches(img1, points1, img2, points2, good_corresponding_points, None, **draw_params)
cv2.imwrite("res15_matches.jpg", result3)


# match only 20 of corresponding points with line and find homography matrix

choose_20_good_point = []
for i, j in corresponding_pairs:
    if i.distance < (0.57645 * j.distance):
        choose_20_good_point.append(i)
img1 = cv2.imread("im03.jpg")
img2 = cv2.imread("im04.jpg")
draw_params = dict(matchColor=(255, 0, 0),
                   singlePointColor=None,
                   matchesMask=None,
                   flags=2)

result3 = cv2.drawMatches(img1, points1, img2, points2, choose_20_good_point, None, **draw_params)
cv2.imwrite("res16.jpg", result3)

# compute homography matrix
corresponding_points_from_picture1 = np.float32(corresponding_points_from_picture1).reshape(-1, 1, 2)
corresponding_points_from_picture2 = np.float32(corresponding_points_from_picture2).reshape(-1, 1, 2)

H, mask = cv2.findHomography(
    corresponding_points_from_picture2,
    corresponding_points_from_picture1,
    cv2.RANSAC,
    5.0,
    maxIters=1000000
)
print(H)
matches_mask = mask.ravel().tolist()

# matching selected corresponding points

img1 = cv2.imread("im03.jpg")
img2 = cv2.imread("im04.jpg")
img2 = cv2.copyMakeBorder(
    img2,
    top=504,
    bottom=504,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT
)
result4 = cv2.hconcat([img1, img2])
corresponding_points_from_picture1 = np.float32(corresponding_points_from_picture1).reshape(-1, 2)
corresponding_points_from_picture2 = np.float32(corresponding_points_from_picture2).reshape(-1, 2)
for i in range(len(corresponding_points_from_picture1)):
    if matches_mask[i]:
        p1 = (int(corresponding_points_from_picture1[i][0]), int(corresponding_points_from_picture1[i][1]))
        p2 = (int(corresponding_points_from_picture2[i][0]) + img1.shape[1], int(corresponding_points_from_picture2[i][1]) + 504)
        result4 = cv2.circle(result4, p1, 2, (0, 0, 255), 2)
        result4 = cv2.circle(result4, p2, 2, (0, 0, 255), 2)
        result4 = cv2.line(result4, p1, p2, (0, 0, 255))
    else:
        p1 = (int(corresponding_points_from_picture1[i][0]), int(corresponding_points_from_picture1[i][1]))
        p2 = (int(corresponding_points_from_picture2[i][0]) + img1.shape[1], int(corresponding_points_from_picture2[i][1]) + 504)
        result4 = cv2.circle(result4, p1, 2, (255, 0, 0), 2)
        result4 = cv2.circle(result4, p2, 2, (255, 0, 0), 2)
        result4 = cv2.line(result4, p1, p2, (255, 0, 0))
cv2.imwrite("res17.jpg", result4)

transpose_matrix = np.array([[1, 0, 3000], [0, 1, 3000], [0, 0, 1]])
H = np.dot(transpose_matrix, H)
# find final picture with homography

img1 = cv2.imread("im03.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("im04.jpg", cv2.IMREAD_COLOR)
dst = cv2.warpPerspective(img2, H, (7 * img1.shape[1], 3 * img1.shape[0]))
cv2.imwrite("res19.jpg", dst)

