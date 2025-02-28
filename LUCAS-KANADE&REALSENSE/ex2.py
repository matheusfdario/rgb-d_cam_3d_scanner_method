import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import matplotlib

import open3d as o3d


matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt

print("Switched to:", matplotlib.get_backend())

# import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API

print("Environment Ready")


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
    #H = np.transpose(AA) * BB
    #H = np.transpose(AA).dot(BB)
    H = np.transpose(AA) @ BB
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

    t = -R @ centroid_A.T + centroid_B.T
    t = t.reshape(3, 1)
    return R, t


ply_path = '/media/matheusfdario/HD/REALSENSE/test/data/EXTRACTED DATA/PLY/ply_1710437557553.69335937500000.ply'
# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/media/matheusfdario/HD/REALSENSE/test/20240314_143232.bag")
profile = pipe.start(cfg)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
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
list_d = [[],[]]

# Show the two frames together:
# images = np.hstack((color, colorized_depth))

# cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('RealSense', images)
# cv2.waitKey(1)
for j in range(2):
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

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

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
        try:
            depth = depth_frame.get_distance(a, b)
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [a, b], depth)
            list_d[j].append(depth_point)
            print(i, a, b, depth_point)
        except:
            print(i, a, b, 'OUT')
    list_x.append(a)
    list_y.append(b)
    #plt.figure(1)
    #plt.plot('frame', img)

    #img = cv2.add(frame, mask)
    #plt.figure(1)
    #plt.plot(img)
    #cv2.imshow('frame', img)
    #cv2.imwrite('../DATASETS/frame.png', img)  # Write frame to output video

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


    #pcd = o3d.io.read_point_cloud(ply_path) # Read the point cloud

    # Visualize the point cloud within open3d
    #o3d.visualization.draw_geometries([pcd])

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format.

    #point_cloud_in_numpy = np.asarray(pcd.points)

    point_cloud_in_numpy = np.asarray(vtx.tolist())
    # max_size = point_cloud_in_numpy.shape[0]
    # view_size = 20000
    # limit_inf = int(max_size/2-view_size/2)
    # limit_sup = int(max_size/2+view_size/2)

    #data = point_cloud_in_numpy[limit_inf:limit_sup]


    point_cloud_valid_only = np.zeros_like(point_cloud_in_numpy)

    limiar = 0.2

    cont = 0

    for i in range(point_cloud_in_numpy.shape[0]):
        if(np.max(np.abs(point_cloud_in_numpy[i]))<limiar):
            point_cloud_valid_only[cont] = point_cloud_in_numpy[i]
            cont = cont + 1

    data = np.zeros([cont+1,3])
    data = point_cloud_valid_only[0:cont]

    # Split the data into x, y, and z arrays
    x = data[::10, 0]
    y = data[::10, 1]
    z = data[::10, 2]
    #x = data[:, 0]
    #y = data[:, 1]
    #z = data[:, 2]
    depro = np.asanyarray(list_d[j])
    xd = depro[:,0]
    yd = depro[:,1]
    zd = depro[:,2]

    # Create a 3D figure
    fig = plt.figure(j)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud data
    ax.scatter(xd,yd,zd,s=10,alpha=1.0)
    ax.scatter(x, y, z, s=1,alpha=0.05,cmap='jet')

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_aspect('equal')
    # Show the plot

# Cleanup:

pipe.stop()
print("Frames Captured")



A = np.asarray(list_d[0])
B = np.asarray(list_d[1])
n = data.shape[0]

Rc, tc = euclidean_transform_3D(A, B)
data_transformed = (Rc@data.T) + np.tile(tc, (1, n))
data_transformed = data_transformed.T

#r0 = Rc@data.T
#t0 = np.tile(tc, (nn0)
#data = (Rc@data.T) + np.tile(tc, data.shape)
#data = data.T

#data1 = (Rc @ data.T)
#data = (Rc @ data.T) + np.transpose(np.tile(tc, (1, n)))
#tc_to_add = np.tile(tc, data.shape[0])
#data = data.T

# Split the data into x, y, and z arrays
x = data_transformed[::10, 0]
y = data_transformed[::10, 1]
z = data_transformed[::10, 2]
#x = data[:, 0]
#y = data[:, 1]
#z = data[:, 2]
depro = np.asanyarray(list_d[j])
xd = depro[:,0]
yd = depro[:,1]
zd = depro[:,2]

# Create a 3D figure
fig = plt.figure(j+1)
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(xd,yd,zd,s=10,alpha=1.0)
ax.scatter(x, y, z, s=1,alpha=0.05,cmap='jet')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
# Show the plot
plt.show()
