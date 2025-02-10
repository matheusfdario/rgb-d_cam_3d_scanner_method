import cv2  # state of the art computer vision algorithms library
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import numpy as np
import matplotlib.pyplot as plt

# path
bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240516_142546 (cópia).bag"
video_name = '/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/VID/video-t.mp4'

# var

cir_size = 3
lines = True
fps = 10.0

D0 = np.zeros([1, 3])
D1 = np.zeros([1, 3])
D2 = np.zeros([1, 3])
D3 = np.zeros([1, 3])

square_corners_str = np.array([[461.8,361.8],[576.4,359.3],[464.4,433.1],[588.5,432.9]])
square_corners_end = np.array([[454.1,77.0],[578.3,62.9],[458.0,151.3],[582.8,141.7]])

rec_corners_str = np.array([[160.7,369.7],[698.0,354.1],[142.2,433.8],[721.4,431.2]])
rec_corners_end = np.array([[129.8,114.8],[711.7,50.7],[138.6,179.0],[716.1,128.9]])

# func



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

pipe = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(config)

#Needed so frames don't get dropped during processing:
#profile.get_device().as_playback().set_real_time(False)

playback = profile.get_device().as_playback()
playback.set_real_time(False)

frame_number = 0

while True:
    print(frame_number)
    try:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        playback.pause()
    except:
        break
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    color = np.asanyarray(color_frame.get_data())
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

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
    color_frame_npy = aligned_color
    print('a')
    if(frame_number==0):
        img_str = color_frame_npy
        print('depro0')
        for i, pix_old in enumerate(square_corners_str):
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
                        D0 = d_points
                    else:
                        D0 = np.vstack((D0, d_points))
                    print(i, a, b, depth_point)
                    deprojection_check = False
                except:
                    print(i, a, b, 'OUT0')
        print('init', D0)
        for i, pix_old in enumerate(rec_corners_str):
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
                        D2 = d_points
                    else:
                        D2 = np.vstack((D2, d_points))
                    print(i, a, b, depth_point)
                    deprojection_check = False
                except:
                    print(i, a, b, 'OUT0')
        print('init', D2)
    else:
        # deprojection in the actual frame
        for i, pix_new in enumerate(square_corners_end):
            c, d = pix_new.ravel()
            c = round(c)
            d = round(d)
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
                print(i, c, d, 'OUT1')
                # deprojection in the actual frame
        for i, pix_new in enumerate(rec_corners_end):
            c, d = pix_new.ravel()
            c = round(c)
            d = round(d)
            try:
                depth = depth_frame.get_distance(c, d)
                depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [c, d], depth)
                # list_d[fi].append(depth_point)
                d_points = np.asarray(depth_point)
                if (i == 0):
                    D3 = d_points
                else:
                    D3 = np.vstack((D3, d_points))
                print(i, c, d, depth_point)
            #     deprojection_check = False
            except:
                print(i, c, d, 'OUT1')
    frame_number += 1  # get number of frames
    playback.resume()
# Cleanup:
pipe.stop()
print("Video Genereted")
img_end = color_frame_npy

plt.figure(0)
plt.imshow(img_str)
plt.scatter(square_corners_str[:,0],square_corners_str[:,1],color='blue')
plt.scatter(rec_corners_str[:,0],rec_corners_str[:,1],color='purple')
plt.figure(1)
plt.imshow(img_end)
plt.scatter(square_corners_end[:,0],square_corners_end[:,1],color='red')
plt.scatter(rec_corners_end[:,0],rec_corners_end[:,1],color='orange')

x0 = D0[:,0]
y0 = D0[:,1]
z0 = D0[:,2]

x1 = D1[:,0]
y1 = D1[:,1]
z1 = D1[:,2]

x2 = D2[:,0]
y2 = D2[:,1]
z2 = D2[:,2]

x3 = D3[:,0]
y3 = D3[:,1]
z3 = D3[:,2]


# Create a 3D figure
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='red')
ax.scatter(x1,y1,z1,s=10.0,color='blue')
ax.scatter(x2,y2,z2,s=10.0,color='purple')
ax.scatter(x3,y3,z3,s=10.0,color='orange')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
ax.set_title('Before TRANS')

A = np.mat(D1)
B = np.mat(D0)
n = A.shape[0]
# recover the transformation
Rc, tc = euclidean_transform_3D(A, B)

A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T
D1T = A_transformed
D0T = D0

A = np.mat(D3)
B = np.mat(D2)
n = A.shape[0]
A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T
D3T = A_transformed
D2T = D2


x0 = D0T[:,0]
y0 = D0T[:,1]
z0 = D0T[:,2]

x1 = D1T[:,0]
y1 = D1T[:,1]
z1 = D1T[:,2]

x2 = D2T[:,0]
y2 = D2T[:,1]
z2 = D2T[:,2]

x3 = D3T[:,0]
y3 = D3T[:,1]
z3 = D3T[:,2]


# Create a 3D figure
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='red')
ax.scatter(x1,y1,z1,s=10.0,color='blue')
ax.scatter(x2,y2,z2,s=10.0,color='purple')
ax.scatter(x3,y3,z3,s=10.0,color='orange')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
ax.set_title('After TRANS')

plt.show()