import numpy as np
import cv2 as cv

#img = cv.imread('/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG/1.png')  # queryImage
img = cv.imread("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/IMG-PIPE/png_Color_1710437557553.99731445312500.png")

#img = cv.imread('home.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

#img=cv.drawKeypoints(gray,kp,img)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imwrite('sift_keypoints.jpg',img)

cv.imwrite('sift_keypoints_1.jpg',img)