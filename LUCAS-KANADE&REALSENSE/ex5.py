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

frame_num = 2

data_list = []
# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/media/matheusfdario/HD/REALSENSE/test/20240314_143232.bag")
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

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

print('a')
# Cleanup:
# pipe.stop()
# print("Frames Captured")

color = np.asanyarray(color_frame.get_data())
# plt.rcParams["axes.grid"] = False
# plt.rcParams['figure.figsize'] = [12, 6]
# plt.imshow(color)
# plt.show()


colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
# plt.imshow(colorized_depth)
# plt.show()

# Create alignment primitive with color as its target stream:
align = rs.align(rs.stream.color)
frameset = align.process(frameset)

# Update color and depth frames:
aligned_depth_frame = frameset.get_depth_frame()
aligned_color_frame = frameset.get_color_frame()
aligned_color = np.asanyarray(aligned_color_frame.get_data())
colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

old_frame = aligned_color

pc = rs.pointcloud()
pc.map_to(aligned_color_frame)
points = pc.calculate(aligned_depth_frame)
vtx = np.asanyarray(points.get_vertices())
tex = np.asanyarray(points.get_texture_coordinates())

# Take first frame and find corners in it
# ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
list_x = []
list_y = []
list_d = [[], []]

# Show the two frames together:
# images = np.hstack((color, colorized_depth))

# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('RealSense', images)
# cv2.waitKey(1)
D0 = np.zeros([1,3])
D1 = np.zeros([1,3])
for j in range(frame_num):
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # print('a')
    # Cleanup:
    # pipe.stop()
    # print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    # plt.rcParams["axes.grid"] = False
    # plt.rcParams['figure.figsize'] = [12, 6]
    # plt.imshow(color)
    # plt.show()

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # plt.imshow(colorized_depth)
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

    # Take first frame and find corners in it
    # ret, old_frame = cap.read()
    frame = aligned_color
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if (p1.shape != p0.shape):
        exit()

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    D0 = D1
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        # if(a <frame.shape[1]):
        #    if(a <frame.shape[0]):
        # Get the  depth value
        #        depth = depth_frame.get_distance(a, b)
        #        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
        #        print(i,a,b,depth_point)
        #    else:
        #        print(i, 'OUT: B')
        # else:
        #    print(i, 'OUT: A')
        deprojection_check = True
        while (deprojection_check):
            try:
                depth = depth_frame.get_distance(a, b)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
                list_d[j].append(depth_point)
                d_points = np.asarray(depth_point)
                if(i==0):
                    D1 = d_points
                else:
                    D1 = np.vstack((D1,d_points))
                print(i, a, b, depth_point)
                deprojection_check = False
            except:
                print(i, a, b, 'OUT')

    list_x.append(a)
    list_y.append(b)
    # plt.figure(1)
    # plt.plot('frame', img)

    # img = cv2.add(frame, mask)
    # plt.figure(1)
    # plt.plot(img)
    # cv2.imshow('frame', img)
    # cv2.imwrite('../DATASETS/frame.png', img)  # Write frame to output video

    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    '''
    # Update previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Show the two frames together:
    images = np.hstack((color, colorized_depth))

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.waitKey(1)
    '''

    # cv2.imshow("Depth Stream", images)
    # plt.imshow(images)
    # plt.show()
    '''
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
    '''

    # pcd = o3d.io.read_point_cloud(ply_path) # Read the point cloud

    # Visualize the point cloud within open3d
    # o3d.visualization.draw_geometries([pcd])

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format.

    # point_cloud_in_numpy = np.asarray(pcd.points)
    vtx_list = vtx.tolist()
    data_list.append(vtx_list)
    point_cloud_in_numpy = np.asarray(vtx_list)
    # max_size = point_cloud_in_numpy.shape[0]
    # view_size = 20000
    # limit_inf = int(max_size/2-view_size/2)
    # limit_sup = int(max_size/2+view_size/2)

    # data = point_cloud_in_numpy[limit_inf:limit_sup]

    point_cloud_valid_only = np.zeros_like(point_cloud_in_numpy)

    limiar = 0.2

    cont = 0

    for i in range(point_cloud_in_numpy.shape[0]):
        if (np.max(np.abs(point_cloud_in_numpy[i])) < limiar):
            point_cloud_valid_only[cont] = point_cloud_in_numpy[i]
            cont = cont + 1

    data = np.zeros([cont + 1, 3])
    data = point_cloud_valid_only[0:cont]
    if(j==0):
        data0 = data
    else:
        data1 = data
    #if (j > 0):
    #     A = np.asarray(list_d[j])  # ESTAVA INVERTIDO
    #     B = np.asarray(list_d[j - 1])
    #     print('A', A, A.shape)
    #     print('B', B, B.shape)
    #     n = A.shape[0]
    #
    #     Rc, tc = euclidean_transform_3D(A, B)
    #     print('tc', tc)
    #     # tc = np.asarray([tc[2],tc[0],tc[1]])
    #     # tc = np.asarray([tc[1], tc[2], tc[0]])
    #     tc = np.flip(tc)
    #     print('tc mod', tc)
    #     A_transformed = (Rc @ A.T) + np.tile(tc, (1, n))
    #     A_transformed = A_transformed.T
    #     n = data.shape[0]
    #     data_transformed = (Rc @ data.T) + np.tile(tc, (1, n))
    #     ajust = np.asarray([0, 0, 0.01]).reshape(3, 1)
    #     # data_transformed = data_transformed + np.tile(ajust, (1, n))
    #     data_transformed = data_transformed.T
    #     rmse = np.sqrt(np.mean(np.square(A_transformed - B)))
    #     print("RMSE:", rmse)
    #     data1 = data_transformed
    #
    #     data_all = np.vstack((data_all, data1))
    #
    # else:
    #     data_all = data
    #     data0 = data
    # data_list.append(data.tolist())
    # # Split the data into x, y, and z arrays
    # x = data[::10, 0]
    # y = data[::10, 1]
    # z = data[::10, 2]
    # # x = data[:, 0]
    # # y = data[:, 1]
    # # z = data[:, 2]
    #
    # # Create a 3D figure
    # fig = plt.figure(j)
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the point cloud data
    # ax.scatter(x, y, z, s=1, alpha=0.05, color="orange")
    #
    # # Set the axis labels
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')
    # Show the plot

# Cleanup:

pipe.stop()
print("Frames Captured")

# A0 = np.asarray(list_d[1])
# B0 = np.asarray(list_d[0])
# n = A0.shape[0]
# Rc, tc = euclidean_transform_3D(A0, B0)
# A = (Rc @ A0.T) + np.tile(tc, (1, n))
# A = A.T
# for w in range(10):
#     rmse = np.sqrt(np.mean(np.square(A - B0)))
#     print("RMSE ", w, ": ", rmse)
#     Rc, tc = euclidean_transform_3D(A, B0)
#     A = (Rc @ A.T) + np.tile(tc, (1, n))
#     A = A.T
#     dist = get_distance_3D(A0, B0)
#     dist_trans = get_distance_3D(A, B0)
#     print(w, dist, dist_trans, dist - dist_trans)
#
#     Rc, tc = euclidean_transform_3D(A, B0)
# Rc, tc = euclidean_transform_3D(A_transformed, B)
# A_transformed2 = (Rc@A_transformed.T) + np.tile(tc, (1, n))
# A_transformed2 = A_transformed2.T
# rmse = np.sqrt(np.mean(np.square(A_transformed2 - B)))
# print ("RMSE2:", rmse)
# r0 = Rc@data.T
# t0 = np.tile(tc, (nn0)
# data = (Rc@data.T) + np.tile(tc, data.shape)
# data = data.T

# data1 = (Rc @ data.T)
# data = (Rc @ data.T) + np.transpose(np.tile(tc, (1, n)))
# tc_to_add = np.tile(tc, data.shape[0])
# data = data.T

# Split the data into x, y, and z arrays
# x = data_all[::10, 0]
# y = data_all[::10, 1]
# z = data_all[::10, 2]

# x = data[:, 0]
# y = data[:, 1]
# z = data[:, 2]
#
# depro0 = np.asanyarray(list_d[0])
# xd0 = depro0[:, 0]
# yd0 = depro0[:, 1]
# zd0 = depro0[:, 2]
# depro1 = np.asanyarray(list_d[1])
# xd1 = depro0[:, 0]
# yd1 = depro0[:, 1]
# zd1 = depro0[:, 2]
#
#
# Create a 3D figure
# fig = plt.figure(j + 1)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
#
# # ax.scatter(xd0, yd0, zd0, s=10.0, alpha=1.0,color="orange")
# # ax.scatter(xd1, yd1, zd1, s=10.0, alpha=1.0,color="red")
# ax.scatter(x, y, z, s=1, alpha=0.05, color="blue")
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# # Show the plot
# #plt.show()
#
# # Split the data into x, y, and z arrays
# depro0 = np.asanyarray(list_d[0][0])
# depro1 = np.asanyarray(list_d[1][0])
#
# depro_max = np.max((depro0.max(),depro1.max()))
# depro_min = np.min((depro0.min(),depro1.min()))
# depro0 = (depro0-depro_min)/(depro_max-depro_min)
# depro1 = (depro1-depro_min)/(depro_max-depro_min)
# ganho = 1000
# depro0 = depro0*ganho
# depro1 = depro1*ganho
# xd0 = depro0[0]
# yd0 = depro0[1]
# zd0 = depro0[2]
#
# xd1 = depro0[0]
# yd1 = depro0[1]
# zd1 = depro0[2]
#
#
# # Create a 3D figure
# fig = plt.figure(j + 2)
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the point cloud data
#
# ax.scatter(xd0, yd0, zd0,color="blue")
# ax.scatter(xd1, yd1, zd1, color="red")
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# # Show the plot
D0p = D0
D1p = D1
D0 = np.mat(D0)
D1 = np.mat(D1)
n = 10
Rc0, tc0 = euclidean_transform_3D(D1, D0)

D1_transformed = (Rc0*D1.T) + np.tile(tc0, (1, n))
D1_transformed = D1_transformed.T
D1_transformed_p = np.asarray(D1_transformed)
# Split the data into x, y, and z arrays
xd0 = D0p[:, 0]
yd0 = D0p[:, 1]
zd0 = D0p[:, 2]

xd1 = D1p[:, 0]
yd1 = D1p[:, 1]
zd1 = D1p[:, 2]


xd1t = D1_transformed_p[:, 0]
yd1t = D1_transformed_p[:, 1]
zd1t = D1_transformed_p[:, 2]

# Create a 3D figure
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xd0,yd0,zd0,color='blue')
ax.scatter(xd1,yd1,zd1,color='red')
for i in range(D0.shape[0]):
    x = np.asarray((xd0[i], xd1[i]))
    y = np.asarray((yd0[i], yd1[i]))
    z = np.asarray((zd0[i], zd1[i]))
    print('xt', x)
    print('yt', y)
    print('zt', z)
    ax.plot(x,y,z, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

# Create a 3D figure
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xd0,yd0,zd0,color='blue')
ax.scatter(xd1t,yd1t,zd1t,color='red')
for i in range(D0.shape[0]):
    xt = np.asarray((xd0[i], xd1t[i]))
    yt = np.asarray((yd0[i], yd1t[i]))
    zt = np.asarray((zd0[i], zd1t[i]))
    print('xt', xt)
    print('yt', yt)
    print('zt', zt)
    ax.plot(xt,yt,zt, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

src = good_old
dst = good_new

# estimate affine transform model using all coordinates
model = AffineTransform()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
model_robust, inliers = ransac(
    (src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=1000
)
outliers = inliers == False

plt.figure(2)
plt.scatter(good_old[:,0],good_old[:,1],color='blue')
plt.scatter(good_new[:,0],good_new[:,1],color='red')
for i in range(good_new.shape[0]):
    x1 = good_old[i,0]
    x2 = good_new[i,0]
    y1 = good_old[i,1]
    y2 = good_new[i, 1]
    xp = np.asarray([good_old[i,0],good_new[i,0]])
    yp = np.asarray([good_old[i,1],good_new[i,1]])
    if(inliers[i]==True):
        plt.plot(xp,yp,'k-',color='green')
    else:
        plt.plot(xp, yp, 'k-', color='orange')
D0p = D0p[inliers==True]
D1p = D1p[inliers==True]
D0 = D0[inliers==True]
D1 = D1[inliers==True]
n = np.count_nonzero(inliers)
Rc, tc = euclidean_transform_3D(D1, D0)

D1_transformed = (Rc*D1.T) + np.tile(tc, (1, n))
D1_transformed = D1_transformed.T
D1_transformed_p = np.asarray(D1_transformed)
# Split the data into x, y, and z arrays
xd0 = D0p[:, 0]
yd0 = D0p[:, 1]
zd0 = D0p[:, 2]

xd1 = D1p[:, 0]
yd1 = D1p[:, 1]
zd1 = D1p[:, 2]


xd1t = D1_transformed_p[:, 0]
yd1t = D1_transformed_p[:, 1]
zd1t = D1_transformed_p[:, 2]

# Create a 3D figure
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xd0,yd0,zd0,color='blue')
ax.scatter(xd1,yd1,zd1,color='red')
for i in range(D0.shape[0]):
    x = np.asarray((xd0[i], xd1[i]))
    y = np.asarray((yd0[i], yd1[i]))
    z = np.asarray((zd0[i], zd1[i]))
    print('xt', x)
    print('yt', y)
    print('zt', z)
    ax.plot(x,y,z, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

# Create a 3D figure
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xd0,yd0,zd0,color='blue')
ax.scatter(xd1t,yd1t,zd1t,color='red')
for i in range(D0.shape[0]):
    xt = np.asarray((xd0[i], xd1t[i]))
    yt = np.asarray((yd0[i], yd1t[i]))
    zt = np.asarray((zd0[i], zd1t[i]))
    print('xt', xt)
    print('yt', yt)
    print('zt', zt)
    ax.plot(xt,yt,zt, color='black')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

plt.show()
