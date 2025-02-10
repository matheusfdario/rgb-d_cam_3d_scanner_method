import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import matplotlib
from scipy.spatial import distance
matplotlib.use('TkAgg', force=True)
print("Switched to:", matplotlib.get_backend())
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

# variables

frame_num = 0
figure_num = 0
lim_perc = 0.25
limiar = 0.2

D0 = np.zeros([1, 3])
D1 = np.zeros([1, 3])

list_x = []
list_y = []
list_d = [[], []]

data_list = []

bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240516_142546.bag"
output = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MERGED/out.npy"

# Setup:/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240426_143549.bag
# Set the playback so it's not done in real-time: https://github.com/IntelRealSense/librealsense/issues/3682#issuecomment-642344385

pipe = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipe.start(config)

playback = profile.get_device().as_playback()
playback.set_real_time(False)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=10,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(20, 20),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    print(frame_num)
    try:
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
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

    cont = 0

    for i in range(point_cloud_in_numpy.shape[0]):
        if (np.max(np.abs(point_cloud_in_numpy[i])) < limiar):
            point_cloud_valid_only[cont] = point_cloud_in_numpy[i]
            cont = cont + 1

    data = np.zeros([cont + 1, 3])
    data = point_cloud_valid_only[0:cont]

    if (frame_num == 0):
        # save pointcloud
        data_old = data
        merged_pointcloud = data_old
        old_frame = aligned_color
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # save in gray scale
        width = old_frame.shape[1]
        height = old_frame.shape[0]
        mask = np.zeros_like(old_gray,dtype=np.uint8)
        # Get the coordinates and dimensions of the detect_box
        x = round(width * lim_perc)
        y = round(height * lim_perc)
        w = width - 2*x
        h = height - 2*y
        # Set the selected rectangle within the mask to white
        mask[y:y + h, x:x + w] = 255
        # corner detection in the first frame
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
        print("data t shape", frame_num, data_new_t.shape)

        # define data old as data new
        data_old = data_new

        merged_pointcloud = np.vstack((merged_pointcloud, data_new_t))

        # redefine old_gray, p0 e D0
        old_gray = frame_gray
        p0 = p0_next

        for i, pix_old in enumerate(p0):    # deprojection is here because it is about the present frame, not the next
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
    color_old = aligned_color
    frame_num += 1
# Cleanup:

pipe.stop()
np.save(output,merged_pointcloud)
print("END")