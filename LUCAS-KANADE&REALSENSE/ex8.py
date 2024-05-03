import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import matplotlib

from skimage import data
from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks, plot_matches
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac
from scipy.spatial import distance
import open3d as o3d

matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt

print("Switched to:", matplotlib.get_backend())

# import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API

print("Environment Ready")


def get_distance_3D(p1, p2):
    squared_dist = np.sum(np.square(p1 - p2))
    distance = np.sqrt(squared_dist)
    return distance


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


def euclidean_transform_3D_2(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)

    # number of points
    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # covariance of datasets
    # H = np.transpose(AA) * BB
    # H = np.transpose(AA).dot(BB)
    H = np.transpose(AA) @ BB
    # matrix decomposition on rotation, scaling and rotation matrices
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    R = Vt.T * U.T
    # print('R', R)
    # prinyt(Vt)
    # print(Vt)
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        print("sign")
        # thanks to @valeriy.krygin to pointing me on a bug here
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        # print('new R', R)

    t = -R @ centroid_A.T + centroid_B.T
    # t = t.T
    t = t.reshape(3, 1)
    return R, t


def test(fig_num):
    R = np.mat(np.random.rand(3, 3))
    t = np.mat(np.random.rand(3, 1))

    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

    if np.linalg.det(R) < 0:
        print('sign')
        Vt[2, :] *= -1
        R = U * Vt
    n = 10

    A = np.mat(np.random.rand(n, 3));
    B = R * A.T + np.tile(t, (1, n))
    B = B.T;

    xa = A[:, 0]
    ya = A[:, 1]
    za = A[:, 2]

    xb = B[:, 0]
    yb = B[:, 1]
    zb = B[:, 2]

    # Create a 3D figure
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data

    ax.scatter(xa, ya, za, color="blue")
    ax.scatter(xb, yb, zb, color="red")

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal')
    # # Show the plot
    # #plt.show()

    # recover the transformation
    Rc, tc = euclidean_transform_3D(A, B)

    A_transformed = (Rc * A.T) + np.tile(tc, (1, n))
    A_transformed = A_transformed.T

    # Find the error
    rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
    print("RMSE:", rmse)


ply_path = '/media/matheusfdario/HD/REALSENSE/test/data/EXTRACTED DATA/PLY/ply_1710437557553.69335937500000.ply'

# variables

frame_num = 4
figure_num = 0

D0 = np.zeros([1, 3])
D1 = np.zeros([1, 3])

list_x = []
list_y = []
list_d = [[], []]

data_list = []

# Setup:/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240426_143549.bag
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240314_143232.bag")
profile = pipe.start(cfg)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.5,
                      minDistance=80,
                      blockSize=10)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 20),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Skip 5 first frames to give the Auto-Exposure time to adjust
# for x in range(5):
#    pipe.wait_for_frames()
for fi in range(frame_num):
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    # pipe.stop()
    # print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    # plt.figure(figure_num)
    # # plt.rcParams["axes.grid"] = False
    # # plt.rcParams['figure.figsize'] = [12, 6]
    # plt.imshow(color)
    # plt.title('RGB %i' % fi)
    # figure_num += 1
    # plt.show()

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # plt.figure(figure_num)
    # plt.imshow(colorized_depth)
    # figure_num += 1
    # plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    aligned_color_frame = frameset.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    aligned_color = np.asanyarray(aligned_color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # get pointcloud with get_vertices
    pc = rs.pointcloud()
    pc.map_to(aligned_color_frame)
    points = pc.calculate(aligned_depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    tex = np.asanyarray(points.get_texture_coordinates())

    vtx_list = vtx.tolist()
    data_list.append(vtx_list)
    point_cloud_in_numpy = np.asarray(vtx_list)

    point_cloud_valid_only = np.zeros_like(point_cloud_in_numpy)

    limiar = 0.2

    cont = 0

    for i in range(point_cloud_in_numpy.shape[0]):
        if (np.max(np.abs(point_cloud_in_numpy[i])) < limiar):
            point_cloud_valid_only[cont] = point_cloud_in_numpy[i]
            cont = cont + 1

    data = np.zeros([cont + 1, 3])
    data = point_cloud_valid_only[0:cont]
    # # plot pointcloud
    # x = data[::100, 0]
    # y = data[::100, 1]
    # z = data[::100, 2]
    # # Create a 3D figure
    # fig = plt.figure(figure_num)
    # figure_num += 1
    # ax = fig.add_subplot(111, projection='3d')
    #
    # #Plot the point cloud data
    # ax.scatter(x,y,z,color='blue')
    # # Set the axis labels
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')
    # ax.set_title('f Pointcloud %i' % frame_num)

    if (fi == 0):
        # save pointcloud
        data_old = data
        merged_pointcloud = data_old
        old_frame = aligned_color
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        width = old_frame.shape[1]
        height = old_frame.shape[0]
        # print("w = %f" % width)
        # print("h = %f" % height)
        # # Initialize the mask with all black pixels
        # mask = [[0] * width] * height
        # mask = np.asarray(mask)
        # mask = mask.astype(np.uint8)
        mask = np.zeros_like(old_gray,dtype=np.uint8)
        print(np.shape(mask))
        # Get the coordinates and dimensions of the detect_box
        lim_perc = 0.1
        x = round(width * lim_perc)
        y = round(height * lim_perc)
        w = width - 2*x
        h = height - 2*y
        #x = 82
        #y = 69
        #w = 35
        #h = 55
        # Set the selected rectangle within the mask to white
        mask[y:y + h, x:x + w] = 255

        #prev = cv.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        #mask = np.zeros_like(first_frame)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
        for i, pix_old in enumerate(p0):
            a, b = pix_old.ravel()
            deprojection_check = True
            while (deprojection_check):
                try:
                    depth = depth_frame.get_distance(a, b)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
                    # list_d[fi].append(depth_point)
                    d_points = np.asarray(depth_point)
                    if (i == 0):
                        D0 = d_points
                    else:
                        D0 = np.vstack((D0, d_points))
                    print(i, a, b, depth_point)
                    deprojection_check = False
                except:
                    print(i, a, b, 'OUT')
        print('init', D0)

        plt.figure(figure_num)
        # plt.rcParams["axes.grid"] = False
        # plt.rcParams['figure.figsize'] = [12, 6]
        # plt.imshow(color)
        plt.title('RGB + CORNER DETECTION %i' % fi)
        # plt.scatter(25, 50, s=500, c='red', marker='o')
        # for j, cord in enumerate(good_old):
        #     print('c1',cord)
        #     plt.scatter(cord[0], cord[1], s=10, c='red', marker='o')
        # for j, cord in enumerate(good_new):
        #     print('c2',cord)
        #     plt.scatter(cord[0], cord[1], s=10, c='blue', marker='o')
        plt.imshow(color)
        good_features = p0[:, 0]
        plt.scatter(good_features[:, 0], good_features[:, 1], color='blue')
        figure_num += 1
        color_old = color
        # dist0 = distance.cdist(D0, data_old)
    else:
        # save pointcloud
        data_new = data
        frame = aligned_color
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0_next = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if (p1.shape != p0.shape):
            exit()

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            st_r = st.reshape(st.shape[0])
            D0 = D0[st_r == 1]

        # deprojection in the actual frame
        for i, pix_new in enumerate(good_new):
            c, d = pix_new.ravel()
            try:
                depth = depth_frame.get_distance(c, d)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, d], depth)
                # list_d[fi].append(depth_point)
                d_points = np.asarray(depth_point)
                if (i == 0):
                    D1 = d_points
                else:
                    D1 = np.vstack((D1, d_points))
                print(i, c, d, depth_point)
            #     deprojection_check = False
            except:
                print(i, c, d, 'OUT')

        # brute force correction of deprojection error

        dist0 = distance.cdist(D0, data_old)
        dist1 = distance.cdist(D1, data_new)
        print('dist0', dist0.shape)
        for i in range(dist0.shape[0]):
            print('i: ', i, " d0_min: ", np.min(dist0[i]), " d1_min: ", np.min(dist1[i]))
        for i in range(dist0.shape[0]):
            D0[i] = data_old[np.argmin(dist0[i])]
            D1[i] = data_new[np.argmin(dist1[i])]
        dist0 = distance.cdist(D0, data_old)
        dist1 = distance.cdist(D1, data_new)
        for i in range(dist0.shape[0]):
            print('i: ', i, " d0_min: ", np.min(dist0[i]), " d1_min: ", np.min(dist1[i]))

        # RANSAC

        # src = good_old
        # dst = good_new
        #
        # # estimate affine transform model using all coordinates
        # model = AffineTransform()
        # model.estimate(src, dst)
        #
        # # robustly estimate affine transform model with RANSAC
        # model_robust, inliers = ransac(
        #     (src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100000
        # )
        # outliers = inliers == False
        #
        # #remove outliers
        #
        # D0 = D0[inliers == True]
        # D1 = D1[inliers == True]

        # convert to numpy matrix
        D0m = np.mat(D0)
        D1m = np.mat(D1)
        data_new_m = np.mat(data_new)
        # calcule euclidean transform 3D
        Rc, tc = euclidean_transform_3D(D1m, D0m)
        # apply defined euclidean transform 3D in the pointcloud
        n = data_new_m.shape[0]
        data_new_t = (Rc * data_new_m.T) + np.tile(tc, (1, n))
        data_new_t = data_new_t.T
        data_new_t = np.asarray(data_new_t)
        print("data t shape", fi, data_new_t.shape)

        # define data old as data new
        data_old = data_new

        # # plot transformed pointcloud
        # x = data[::100, 0]
        # y = data[::100, 1]
        # z = data[::100, 2]
        # # Create a 3D figure
        # fig = plt.figure(figure_num)
        # figure_num += 1
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # Plot the point cloud data
        # ax.scatter(x, y, z, color='red')
        # # Set the axis labels
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_aspect('equal')
        # ax.set_title('f Trans Pointcloud %i' % frame_num)
        merged_pointcloud = np.vstack((merged_pointcloud, data_new_t))
        # dist0 = distance.cdist(D0, data_old)
        # dist1 = distance.cdist(D1, data1)

        # redefine old_gray, p0 e D0
        old_gray = frame_gray
        p0 = p0_next

        for i, pix_old in enumerate(p0):
            a, b = pix_old.ravel()
            deprojection_check = True
            while (deprojection_check):
                try:
                    depth = depth_frame.get_distance(a, b)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
                    # list_d[fi].append(depth_point)
                    d_points = np.asarray(depth_point)
                    if (i == 0):
                        D0 = d_points
                    else:
                        D0 = np.vstack((D0, d_points))
                    print(i, a, b, depth_point)
                    deprojection_check = False
                except:
                    print(i, a, b, 'OUT')
        print('init', D0)

        list_x.append(a)
        list_y.append(b)

        # plot

        plt.figure(figure_num)
        # plt.rcParams["axes.grid"] = False
        # plt.rcParams['figure.figsize'] = [12, 6]
        # plt.imshow(color)
        plt.title('RGB + CORNER DETECTION %i' % fi)
        # plt.scatter(25, 50, s=500, c='red', marker='o')
        # for j, cord in enumerate(good_old):
        #     print('c1',cord)
        #     plt.scatter(cord[0], cord[1], s=10, c='red', marker='o')
        # for j, cord in enumerate(good_new):
        #     print('c2',cord)
        #     plt.scatter(cord[0], cord[1], s=10, c='blue', marker='o')
        color_plt = np.concatenate((color_old, color), axis=1)
        plt.imshow(color_plt)
        #good_features = p0[:, 0]
        good_features = good_old
        plt.scatter(good_features[:, 0], good_features[:, 1], color='blue')
        correction = np.asarray([color_old.shape[1],0])
        #good_features_lk = p1[:, 0] + correction
        good_features_lk = good_new + correction
        plt.scatter(good_features_lk[:, 0], good_features_lk[:, 1], color='red')
        figure_num += 1
        plt.figure(figure_num)
        plt.scatter(good_old[:, 0], good_old[:, 1], color='blue')
        plt.scatter(good_new[:, 0], good_new[:, 1], color='red')
        for i in range(good_new.shape[0]):
            x1 = good_old[i, 0]
            x2 = good_new[i, 0]
            y1 = good_old[i, 1]
            y2 = good_new[i, 1]
            xp = np.asarray([good_old[i, 0], good_new[i, 0]])
            yp = np.asarray([good_old[i, 1], good_new[i, 1]])
            # if (inliers[i] == True):
            #     plt.plot(xp, yp, 'k-', color='green')
            # else:
            #     plt.plot(xp, yp, 'k-', color='orange')
            plt.plot(xp, yp, 'k-', color='black')
        figure_num += 1

    # plot transformed merged pointcloud
    x = merged_pointcloud[::100, 0]
    y = merged_pointcloud[::100, 1]
    z = merged_pointcloud[::100, 2]
    # Create a 3D figure
    fig = plt.figure(figure_num)
    figure_num += 1
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    ax.scatter(x, y, z, color='green')
    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal')
    ax.set_title('Merged Pointcloud %i' % fi)

# Cleanup:

pipe.stop()
print("Frames Captured")

# print('dist0',dist0.shape)
# for i in range(dist0.shape[0]):
#     print('i: ',i," d0_min: ",np.min(dist0[i])," d1_min: ",np.min(dist1[i]))
# for i in range(dist0.shape[0]):
#     D0[i] = data0[np.argmin(dist0[i])]
#     D1[i] = data1[np.argmin(dist1[i])]
# dist0 = distance.cdist(D0,data0)
# dist1 = distance.cdist(D1, data1)
# for i in range(dist0.shape[0]):
#     print('i: ',i," d0_min: ",np.min(dist0[i])," d1_min: ",np.min(dist1[i]))
#
# D0p = D0
# D1p = D1
#
# D0 = np.mat(D0)
# D1 = np.mat(D1)
# n = 10
# Rc0, tc0 = euclidean_transform_3D(D1, D0)
#
# D1_transformed = (Rc0*D1.T) + np.tile(tc0, (1, n))
# D1_transformed = D1_transformed.T
# D1_transformed_p = np.asarray(D1_transformed)
# # Split the data into x, y, and z arrays
# xd0 = D0p[:, 0]
# yd0 = D0p[:, 1]
# zd0 = D0p[:, 2]
#
# xd1 = D1p[:, 0]
# yd1 = D1p[:, 1]
# zd1 = D1p[:, 2]
#
#
# xd1t = D1_transformed_p[:, 0]
# yd1t = D1_transformed_p[:, 1]
# zd1t = D1_transformed_p[:, 2]
#
# # Create a 3D figure
# fig = plt.figure(0)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
# ax.scatter(xd0,yd0,zd0,color='blue')
# ax.scatter(xd1,yd1,zd1,color='red')
# for i in range(D0.shape[0]):
#     x = np.asarray((xd0[i], xd1[i]))
#     y = np.asarray((yd0[i], yd1[i]))
#     z = np.asarray((zd0[i], zd1[i]))
#     print('xt', x)
#     print('yt', y)
#     print('zt', z)
#     ax.plot(x,y,z, color='black')
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# # Create a 3D figure
# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
# ax.scatter(xd0,yd0,zd0,color='blue')
# ax.scatter(xd1t,yd1t,zd1t,color='red')
# for i in range(D0.shape[0]):
#     xt = np.asarray((xd0[i], xd1t[i]))
#     yt = np.asarray((yd0[i], yd1t[i]))
#     zt = np.asarray((zd0[i], zd1t[i]))
#     print('xt', xt)
#     print('yt', yt)
#     print('zt', zt)
#     ax.plot(xt,yt,zt, color='black')
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# src = good_old
# dst = good_new
#
# # estimate affine transform model using all coordinates
# model = AffineTransform()
# model.estimate(src, dst)
#
# # robustly estimate affine transform model with RANSAC
# model_robust, inliers = ransac(
#     (src, dst), AffineTransform, min_samples=3, residual_threshold=3, max_trials=1000
# )
# outliers = inliers == False
#
# plt.figure(2)
# plt.scatter(good_old[:,0],good_old[:,1],color='blue')
# plt.scatter(good_new[:,0],good_new[:,1],color='red')
# for i in range(good_new.shape[0]):
#     x1 = good_old[i,0]
#     x2 = good_new[i,0]
#     y1 = good_old[i,1]
#     y2 = good_new[i, 1]
#     xp = np.asarray([good_old[i,0],good_new[i,0]])
#     yp = np.asarray([good_old[i,1],good_new[i,1]])
#     if(inliers[i]==True):
#         plt.plot(xp,yp,'k-',color='green')
#     else:
#         plt.plot(xp, yp, 'k-', color='orange')
# D0p = D0p[inliers==True]
# D1p = D1p[inliers==True]
# D0 = D0[inliers==True]
# D1 = D1[inliers==True]
# n = np.count_nonzero(inliers)
# Rc, tc = euclidean_transform_3D(D1, D0)
#
# D1_transformed = (Rc*D1.T) + np.tile(tc, (1, n))
# D1_transformed = D1_transformed.T
# D1_transformed_p = np.asarray(D1_transformed)
# # Split the data into x, y, and z arrays
# xd0 = D0p[:, 0]
# yd0 = D0p[:, 1]
# zd0 = D0p[:, 2]
#
# xd1 = D1p[:, 0]
# yd1 = D1p[:, 1]
# zd1 = D1p[:, 2]
#
#
# xd1t = D1_transformed_p[:, 0]
# yd1t = D1_transformed_p[:, 1]
# zd1t = D1_transformed_p[:, 2]
#
# # Create a 3D figure
# fig = plt.figure(3)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
# ax.scatter(xd0,yd0,zd0,color='blue')
# ax.scatter(xd1,yd1,zd1,color='red')
# for i in range(D0.shape[0]):
#     x = np.asarray((xd0[i], xd1[i]))
#     y = np.asarray((yd0[i], yd1[i]))
#     z = np.asarray((zd0[i], zd1[i]))
#     print('xt', x)
#     print('yt', y)
#     print('zt', z)
#     ax.plot(x,y,z, color='black')
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# # Create a 3D figure
# fig = plt.figure(4)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
# ax.scatter(xd0,yd0,zd0,color='blue')
# ax.scatter(xd1t,yd1t,zd1t,color='red')
# for i in range(D0.shape[0]):
#     xt = np.asarray((xd0[i], xd1t[i]))
#     yt = np.asarray((yd0[i], yd1t[i]))
#     zt = np.asarray((zd0[i], zd1t[i]))
#     print('xt', xt)
#     print('yt', yt)
#     print('zt', zt)
#     ax.plot(xt,yt,zt, color='black')
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# #Split the data into x, y, and z arrays
# data_all = data0
# data1m = np.mat(data1)
# n = data1.shape[0]
# data1mt0 = (Rc0*data1m.T) + np.tile(tc0, (1, n))
# data1mt0 = data1mt0.T
# data1t0 = np.asarray(data1mt0)
# data_all = np.vstack((data_all,data1t0))
# xf0 = data_all[::10, 0]
# yf0 = data_all[::10, 1]
# zf0 = data_all[::10, 2]
# data_all = data0
# data1mt1 = (Rc*data1m.T) + np.tile(tc, (1, n))
# data1mt1 = data1mt1.T
# data1t1 = np.asarray(data1mt1)
# data_all = np.vstack((data_all,data1t1))
# xf1 = data_all[::10, 0]
# yf1 = data_all[::10, 1]
# zf1 = data_all[::10, 2]
#
# #Create a 3D figure
# fig = plt.figure(5)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
#
# # ax.scatter(xd0, yd0, zd0, s=10.0, alpha=1.0,color="orange")
# # ax.scatter(xd1, yd1, zd1, s=10.0, alpha=1.0,color="red")
# ax.scatter(xf0, yf0, zf0, s=1, alpha=0.05, color="red")
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# #Create a 3D figure
# fig = plt.figure(6)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
#
# # ax.scatter(xd0, yd0, zd0, s=10.0, alpha=1.0,color="orange")
# # ax.scatter(xd1, yd1, zd1, s=10.0, alpha=1.0,color="red")
# ax.scatter(xf1, yf1, zf1, s=1, alpha=0.05, color='red')
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
#
# plt.show()
# dist0 = distance.cdist(D0p,data0)
# dist1 = distance.cdist(D1p, data1)
# for i in range(dist0.shape[0]):
#     print('i: ',i," d0_min: ",np.min(dist0[i])," d1_min: ",np.min(dist1[i]))