import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def feature_matching(features0, features1):
    matches = [] # good matches as per Lowe's ratio test
    if(features0.des is not None and len(features0.des) > 2):
        all_matches = flann.knnMatch( \
            features0.des, features1.des, k=2)
        try:
            for m,n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if(len(matches) > MIN_MATCHES):
            features0.matched_pts = np.float32( \
                [ features0.kps[m.queryIdx].pt for m in matches ] \
                ).reshape(-1,1,2)
            features1.matched_pts = np.float32( \
                [ features1.kps[m.trainIdx].pt for m in matches ] \
                ).reshape(-1,1,2)
    return matches

img1 = cv.imread("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG-PIPE/png_Color_1710437557553.99731445312500.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG-PIPE/png_Color_1710437557587.33837890625000.png", cv.IMREAD_GRAYSCALE)

#img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)  # queryImage
#img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

#matches = feature_matching(des1, des2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]
        print(m.queryIdx)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3, ), plt.show()

