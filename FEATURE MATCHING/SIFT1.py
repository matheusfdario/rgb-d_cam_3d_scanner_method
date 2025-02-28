import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
#img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE) # queryImage
#img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE) # trainImage


#img1 = cv.imread("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG-PIPE/png_Color_1710437557553.99731445312500.png", cv.IMREAD_GRAYSCALE)
#img2 = cv.imread("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG-PIPE/png_Color_1710437557587.33837890625000.png", cv.IMREAD_GRAYSCALE)

img1 = cv.imread('/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG/1.png', cv.IMREAD_GRAYSCALE)  # queryImage
img2 = cv.imread('/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG/200.png', cv.IMREAD_GRAYSCALE)  # trainImage

def centeroidnp(arr):
    arr = arr.reshape(4, 2)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    centroid = np.asarray([sum_x / length, sum_y / length])
    return centroid
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(img2, [np.int32(dst)], True, 200, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
singlePointColor = None,
matchesMask = matchesMask, # draw only inliers
flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()
matchesMask_np = np.asarray(matchesMask)
src_pts_mask = src_pts[matchesMask_np==1]
dst_pts_mask = dst_pts[matchesMask_np==1]