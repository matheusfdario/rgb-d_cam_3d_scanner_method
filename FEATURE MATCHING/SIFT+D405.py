import cv2 as cv  # state of the art computer vision algorithms library
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import variation



from matplotlib import pyplot as plt

# path
bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240524_173706_ok.bag"
video_name = '/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/VID/video-t.mp4'

# var
MIN_MATCH_COUNT = 10
cir_size = 3
lines = True
fps = 10.0
get_depth = False
first_frame = True
play_playback = True
var_max = 0.15
dist_min = 25

D0 = np.zeros([1, 3])
D1 = np.zeros([1, 3])
D2 = np.zeros([1, 3])
D3 = np.zeros([1, 3])

square_corners_str = np.array([[461.8,361.8],[576.4,359.3],[464.4,433.1],[588.5,432.9]])
square_corners_end = np.array([[454.1,77.0],[578.3,62.9],[458.0,151.3],[582.8,141.7]])

rec_corners_str = np.array([[160.7,369.7],[698.0,354.1],[142.2,433.8],[721.4,431.2]])
rec_corners_end = np.array([[129.8,114.8],[711.7,50.7],[138.6,179.0],[716.1,128.9]])

# func

# ==============================================================================
#                                                                     VIZ_MAYAVI
# ==============================================================================
# def viz_mayavi(points, vals="distance"):
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of point
#     # r = lidar[:, 3]  # reflectance value of point
#     d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
#
#     # Plot using mayavi -Much faster and smoother than matplotlib
#     import mayavi.mlab
#
#     if vals == "height":
#         col = z
#     else:
#         col = d
#
#     fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
#     mayavi.mlab.points3d(x, y, z,
#                          col,          # Values used for Color
#                          mode="point",
#                          colormap='spectral', # 'bone', 'copper', 'gnuplot'
#                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
#                          figure=fig,
#                          )
#     mayavi.mlab.show()

def reject_outliers(data, m = 125.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

def centeroidnp(arr):
    arr = arr.reshape(4, 2)
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    centroid = np.asarray([sum_x / length, sum_y / length])
    return centroid
def feature_matching(image1,image2,fn):
    img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

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
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)               #find Homography just to use RANSAC isn't optimazed, but it`s implemented!
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        pts_cent = centeroidnp(pts)
        dst_cent = centeroidnp(dst)
        dist_var = np.linalg.norm(dst_cent - pts_cent)

        img2 = cv.polylines(img2, [np.int32(dst)], True, 200, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)
    #plt.figure(fn)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #plt.imshow(img3, 'gray'),plt.show()
    matchesMask_np = np.asarray(matchesMask)
    src_pts_mask = src_pts[matchesMask_np==1]
    dst_pts_mask = dst_pts[matchesMask_np==1]
    num_mat =len(src_pts_mask)
    valid_matches = np.stack((src_pts_mask[:, 0], dst_pts_mask[:, 0]), axis=0)
    #dist_var = np.mean(reject_outliers(np.diag(distance.cdist(valid_matches[0], valid_matches[1]))))
    disp_var = variation(np.diag(distance.cdist(valid_matches[0], valid_matches[1])))
    return valid_matches,dist_var,disp_var, num_mat

# brute force correction of deprojection error
def brute_force_pointcloud_correction(pointset,pointcloud):
    dist = distance.cdist(pointset, pointcloud)
    pointset_corrected = pointcloud[dist.argmin(axis=1)]
    return pointset_corrected

# deprojection of a seto os points
def pointset_deprojection(pixels,depth_frama,depth_intrin):
    for i, pixel in enumerate(square_corners_str):
        a, b = pix_old.ravel()
        a = round(a)
        b = round(b)
        print(a, b)
        deprojection_check = True
        while (deprojection_check):
            try:
                depth = depth_frame.get_distance(a, b)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
                # list_d[fi].append(depth_point)
                d_points = np.asarray(depth_point)
                if (i == 0):
                    D = d_points
                else:
                    D = np.vstack((D0, d_points))
                print(i, a, b, depth_point)
                deprojection_check = False
            except:
                print(i, a, b, 'OUT0')
    return D
def filter_invalid_3D_points(pointset0,pointset1,threshold):
    distance0 = np.linalg.norm(pointset0, axis=1)
    distance1 = np.linalg.norm(pointset1, axis=1)
    mask = np.ones_like(distance0)
    mask = np.where(distance0>threshold, mask,0)
    mask = np.where(distance1>threshold, mask,0)
    pointset0_filtered = pointset0[mask>0]
    pointset1_filtered = pointset1[mask>0]
    return pointset0_filtered,pointset1_filtered
def euclidean_transform_3D(A, B):
    '''
            A,B - Nx3 matrix
            return:
                R - 3x3 rotation matrix
                t = 3x1 column vector
        '''
    print('eu', A.shape, B.shape)
    assert len(A) == len(B)

    # number of points
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # covariance of datasets
    H = np.transpose(AA) * BB

    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    R = Vt.T * U.T
    print('R', R)
    # prinyt(Vt)
    print(Vt)
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        print("sign")
        # thanks to @valeriy.krygin to pointing me on a bug here
        Vt[2, :] *= -1
        R = Vt.T * U.T
        print('new R', R)

    t = -R * centroid_A.T + centroid_B.T

    return R, t

# Setup:/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240426_143549.bag
# Set the playback so it's not done in real-time: https://github.com/IntelRealSense/librealsense/issues/3682#issuecomment-642344385

dist_list = []
disp_list = []
num_matches_list = []
pointclouds = []
pointclouds_T = []

sel_frames = [0] #for generalization
fig_number = 0
while(play_playback):
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipe.start(config)

    #Needed so frames don't get dropped during processing:
    #profile.get_device().as_playback().set_real_time(False)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    total_time = playback.get_duration()
    total_time_nano = total_time.microseconds*1000
    frame_number = 0
    time_now = playback.get_position()
    dist = 0
    disp = 3
    combined_frames = 0
    while True:
        if(get_depth):
            print(frame_number)
        else:
            print(frame_number,dist)
        try:
            # Store next frameset for later processing:
            frameset = pipe.wait_for_frames()
            playback.pause()
        except:
            if(get_depth):
                play_playback = False
            else:
                get_depth = True
            break
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        color = np.asanyarray(color_frame.get_data())
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()
        aligned_color_frame = frameset.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        aligned_color = np.asanyarray(aligned_color_frame.get_data())
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        color_frame_npy = aligned_color

        if(get_depth):
            # get pointcloud with get_vertices
            pc = rs.pointcloud()
            pc.map_to(aligned_color_frame)
            points = pc.calculate(aligned_depth_frame)
            vtx = np.asanyarray(points.get_vertices())
            tex = np.asanyarray(points.get_texture_coordinates())

            vtx_list = vtx.tolist()
            #data_list.append(vtx_list)
            point_cloud_in_numpy = np.asarray(vtx_list)

            point_cloud_valid_only = np.zeros_like(point_cloud_in_numpy)

            # 3D filter for pointcloud

            radius = 0.5

            pointcloud_X = point_cloud_in_numpy[:,0]
            pointcloud_Y = point_cloud_in_numpy[:,1]
            pointcloud_Z = point_cloud_in_numpy[:,2]
            pointcloud_mask = 2*(pointcloud_X**2) + 2*(pointcloud_Y**2) + 2*(pointcloud_Z**2) <= 3*(radius**2)

            data = point_cloud_in_numpy[pointcloud_mask]

            print('a')
            if(frame_number==0):
                data_str = data
                data0 = data
                data_merged = data_str
                pointclouds.append(data_str)
                pointclouds_T.append(data_str)
                img_str = color_frame_npy
                print('depro0')
                matches_pair = matches_list[0]
                points0 = matches_pair[0]
                for i, pix_old in enumerate(points0):
                    a, b = pix_old.ravel()
                    a = round(a)
                    b = round(b)
                    print(a,b)
                    deprojection_check = True
                    while (deprojection_check):
                        try:
                            depth = depth_frame.get_distance(a, b)
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
                            # list_d[fi].append(depth_point)
                            d_points = np.asarray(depth_point)
                            if (i == 0):
                                P0 = d_points
                            else:
                                P0 = np.vstack((P0, d_points))
                            print(i, a, b, depth_point)
                            deprojection_check = False
                        except:
                            print(i, a, b, 'OUT0')
                print('init', P0)

                # brute force correction of deprojection error
                P0 = brute_force_pointcloud_correction(P0,data)

            else:
                if frame_number in sel_frames:
                    print('merge ',frame_number)
                    data_end = data
                    data1 = data
                    # deprojection in the actual frame
                    matches_pair = matches_list[combined_frames]
                    points1 = matches_pair[1] #dst

                    for i, pix_new in enumerate(points1):
                        c, d = pix_new.ravel()
                        c = round(c)
                        d = round(d)
                        print(i,c,d)
                        deprojection_check = True
                        while(deprojection_check):
                            try:
                                depth = depth_frame.get_distance(c, d)
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, d], depth)
                                # list_d[fi].append(depth_point)
                                d_points = np.asarray(depth_point)
                                print('depro',d_points)
                                if (i == 0):
                                    P1 = d_points
                                else:
                                    P1 = np.vstack((P1, d_points))
                                print(i, c, d, depth_point)
                                deprojection_check = False
                            except:
                                print(i, c, d, 'OUT1')
                    # brute force correction of deprojection error
                    P1 = brute_force_pointcloud_correction(P1, data)
                    combined_frames += 1
                    if(combined_frames<len(matches_list)):
                        matches_pair = matches_list[combined_frames]
                        points0_next = matches_pair[0]
                        for i, pix_new in enumerate(points0_next):
                            c, d = pix_new.ravel()
                            c = round(c)
                            d = round(d)
                            try:
                                depth = depth_frame.get_distance(c, d)
                                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, d], depth)
                                # list_d[fi].append(depth_point)
                                d_points = np.asarray(depth_point)
                                if (i == 0):
                                    P0_next = d_points
                                else:
                                    P0_next = np.vstack((P0_next, d_points))
                                print(i, c, d, depth_point)
                            #     deprojection_check = False
                            except:
                                print(i, c, d, 'OUT1')
                        # brute force correction of deprojection error
                        P0_next = brute_force_pointcloud_correction(P0_next, data)


                    A = np.mat(P1)
                    B = np.mat(P0)
                    n = A.shape[0]
                    # recover the transformation
                    Rc, tc = euclidean_transform_3D(A, B)
                    A = np.mat(data1)
                    B = np.mat(data0)
                    n = A.shape[0]
                    A_transformed = (Rc * A.T) + np.tile(tc, (1, n))
                    A_transformed = A_transformed.T
                    data_end_T = A_transformed
                    #data_str_T = data_str
                    data_merged = np.vstack((data_merged, data_end_T))
                    P0 = P0_next
                    data0 = data1


        else:
            if(frame_number==0):
                matches_list = []
                img_last = color_frame_npy
            else:
                img_now = color_frame_npy
                matches, dist, disp, num_matches = feature_matching(img_last,img_now)
                if(dist>dist_min):
                    if(disp<var_max):
                        sel_frames.append(frame_number)
                        matches_list.append(matches)
                        img_last = color_frame_npy
                dist_list.append(dist)
                disp_list.append(disp)
                num_matches_list.append(num_matches)

        frame_number += 1  # get number of frames
        playback.resume()

    # Cleanup:
    pipe.stop()
    print("Video Genereted")
img_end = color_frame_npy

# plt.figure(0)
# plt.imshow(img_str)
# plt.scatter(square_corners_str[:,0],square_corners_str[:,1],color='blue')
# plt.scatter(rec_corners_str[:,0],rec_corners_str[:,1],color='purple')
# plt.figure(1)
# plt.imshow(img_end)
# plt.scatter(square_corners_end[:,0],square_corners_end[:,1],color='red')
# plt.scatter(rec_corners_end[:,0],rec_corners_end[:,1],color='orange')
#
# x0 = D0[:,0]
# y0 = D0[:,1]
# z0 = D0[:,2]
#
# x1 = D1[:,0]
# y1 = D1[:,1]
# z1 = D1[:,2]
#
# x2 = D2[:,0]
# y2 = D2[:,1]
# z2 = D2[:,2]
#
# x3 = D3[:,0]
# y3 = D3[:,1]
# z3 = D3[:,2]
#
#
# # Create a 3D figure
# fig = plt.figure(3)
# ax = fig.add_subplot(111, projection='3d')
#
# #Plot the point cloud data
# ax.scatter(x0,y0,z0,s=10.0,color='red')
# ax.scatter(x1,y1,z1,s=10.0,color='blue')
# ax.scatter(x2,y2,z2,s=10.0,color='purple')
# ax.scatter(x3,y3,z3,s=10.0,color='orange')
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.set_title('Before TRANS')
#
# xs = data_str[::1000, 0]
# ys = data_str[::1000, 1]
# zs = data_str[::1000, 2]
#
# xe = data_end[::1000, 0]
# ye = data_end[::1000, 1]
# ze = data_end[::1000, 2]
#
# # Create a 3D figure
# fig = plt.figure(4)
# ax = fig.add_subplot(111, projection='3d')
#
# #Plot the point cloud data
# ax.scatter(xs,ys,zs,s=10.0,color='red')
# ax.scatter(xe,ye,ze,s=10.0,color='blue')
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.set_title('Pointcloud Before TRANS')
#
# A = np.mat(D1)
# B = np.mat(D0)
# n = A.shape[0]
# # recover the transformation
# Rc, tc = euclidean_transform_3D(A, B)
#
# A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
# A_transformed = A_transformed.T
# D1T = A_transformed
# D0T = D0
#
# A = np.mat(D3)
# B = np.mat(D2)
# n = A.shape[0]
# A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
# A_transformed = A_transformed.T
# D3T = A_transformed
# D2T = D2
#
# x0 = D0T[:,0]
# y0 = D0T[:,1]
# z0 = D0T[:,2]
#
# x1 = D1T[:,0]
# y1 = D1T[:,1]
# z1 = D1T[:,2]
#
# x2 = D2T[:,0]
# y2 = D2T[:,1]
# z2 = D2T[:,2]
#
# x3 = D3T[:,0]
# y3 = D3T[:,1]
# z3 = D3T[:,2]
#
#
# # Create a 3D figure
# fig = plt.figure(5)
# ax = fig.add_subplot(111, projection='3d')
#
# #Plot the point cloud data
# ax.scatter(x0,y0,z0,s=10.0,color='red')
# ax.scatter(x1,y1,z1,s=10.0,color='blue')
# ax.scatter(x2,y2,z2,s=10.0,color='purple')
# ax.scatter(x3,y3,z3,s=10.0,color='orange')
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.set_title('After TRANS')
#
# A = np.mat(data_end)
# B = np.mat(data_str)
# n = A.shape[0]
# A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
# A_transformed = A_transformed.T
# data_end_T = A_transformed
# data_str_T = data_str
#
# xst = data_str_T[::1000, 0]
# yst = data_str_T[::1000, 1]
# zst = data_str_T[::1000, 2]
#
# xet = data_end_T[::1000, 0]
# yet = data_end_T[::1000, 1]
# zet = data_end_T[::1000, 2]
#
# # Create a 3D figure
# fig = plt.figure(6)
# ax = fig.add_subplot(111, projection='3d')
#
# #Plot the point cloud data
# ax.scatter(xst,yst,zst,s=10.0,color='red')
# ax.scatter(xet,yet,zet,s=10.0,color='blue')
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.set_title('Poincloud After TRANS')
#
# plt.show()

xm = data_merged[::1000, 0]
ym = data_merged[::1000, 1]
zm = data_merged[::1000, 2]


# Create a 3D figure
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(xm,ym,zm,s=10.0,color='red')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
ax.set_title('Poincloud After TRANS')

# img1 = cv.cvtColor(img_str, cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img_end, cv.COLOR_BGR2GRAY)
#
# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     #h, w = img1.shape
#     #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     #dst = cv.perspectiveTransform(pts, M)
#     #img2 = cv.polylines(img2, [np.int32(dst)], True, 200, 3, cv.LINE_AA)
# else:
#     print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#     matchesMask = None
# #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
# #singlePointColor = None,
# #matchesMask = matchesMask, # draw only inliers
# #flags = 2)
# #plt.figure(7)
# #img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# #plt.imshow(img3, 'gray'),plt.show()
# matchesMask_np = np.asarray(matchesMask)
# src_pts_mask = src_pts[matchesMask_np==1]
# dst_pts_mask = dst_pts[matchesMask_np==1]
# valid_matches = np.stack((src_pts_mask[:,0],dst_pts_mask[:,0]),axis=0)
plt.figure(1)
dist_np = np.asarray(dist_list)
plt.plot(range(len(dist_np)),dist_np)
plt.figure(2)
disp_np = np.asarray(disp_list)
plt.plot(range(len(disp_np)),disp_np)
plt.figure(3)
num_matches_np = np.asarray(num_matches_list)
plt.plot(range(len(num_matches_np)),num_matches_np)
plt.show()
# plt.figure(9)
# vec = reject_outliers(np.diag(distance.cdist(valid_matches[0], valid_matches[1])))
# plt.hist(vec)