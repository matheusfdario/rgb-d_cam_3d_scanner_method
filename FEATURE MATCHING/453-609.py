import cv2 as cv  # state of the art computer vision algorithms library
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import variation
import datetime,os
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
# path
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/CIR/20240426_145334.bag" # inner pipe
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20240726_190520.bag"
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241114_163807.bag"
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241119_102249.bag" # dani pipe ok
bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241119_135757.bag" # spool slice a ok
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241119_162210.bag" # spool slice b (black)
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/" #tati pipe
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241205_180427.bag" #hex tube 1
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241210_144344.bag" #hex tube 2
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241216_180242.bag"
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241211_150942.bag" # pato 1
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241216_180242.bag" # meias cana
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241217_114828.bag" #test spray 3d + gis
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241217_151702.bag" # test shampoo seco
#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20250129_190853.bag" # test hex 3

#bag_file = "/home/matheusfdario/Documentos/PC-BAG/20241113_143343.bag" #real ok 5 fps
#bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240726_190520.bag"            #real big
#bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240507_170957.bag"             #real small
#bag_file = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240516_142546 (cópia).bag"    #test
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
video_name = '/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/VID/video-'+ timestamp + '.mp4'
pc_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/" + timestamp
os.makedirs(pc_path)
img_path = pc_path + "/IMG"
os.makedirs(img_path)
trans_path = pc_path + "/TRANS"
os.makedirs(trans_path)
# var
MAX_MATCH_COUNT = 100
MIN_MATCH_COUNT = 10
MIN_DIST = 10 #PIXELS
cir_size = 3
cir_size = 3
lines = True
fps = 10.0
get_depth = False
first_frame = True
play_playback = True
pair_zero_flag = True
var_max = 0.15
dist_min = 25
dist_var = 0
D0 = np.zeros([1, 3])
D1 = np.zeros([1, 3])
D2 = np.zeros([1, 3])
D3 = np.zeros([1, 3])
square_corners_str = np.array([[461.8,361.8],[576.4,359.3],[464.4,433.1],[588.5,432.9]])
square_corners_end = np.array([[454.1,77.0],[578.3,62.9],[458.0,151.3],[582.8,141.7]])
rec_corners_str = np.array([[160.7,369.7],[698.0,354.1],[142.2,433.8],[721.4,431.2]])
rec_corners_end = np.array([[129.8,114.8],[711.7,50.7],[138.6,179.0],[716.1,128.9]])
# func
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
def plot3D(f_num, path):
    pc_list = sorted(os.listdir(path))
    pc_list = pc_list[:-2]
    # Create a 3D figure
    fig = plt.figure(f_num)
    ax = fig.add_subplot(111, projection='3d')

    # Set the axis labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Pointclouds 2')

    # Initialize lists to collect all point coordinates
    all_x, all_y, all_z = [], [], []

    colors = cm.rainbow(np.linspace(0, 1, len(pc_list)))
    for i, npc in enumerate(pc_list):
        load_path = os.path.join(path, npc)
        print(load_path)
        pc = np.load(load_path)
        x = pc[::1000, 0]
        y = pc[::1000, 1]
        z = pc[::1000, 2]

        # Collect the points for later adjustment of the aspect ratio
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)

        # Plot the point cloud data
        ax.scatter(x, y, z, s=1.0, color=colors[i], alpha=0.75)

    # Calculate the ranges for each axis
    max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5

    # Set the limits for each axis to make them equal
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
# Setup:/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/20240426_143549.bag
# Set the playback so it's not done in real-time: https://github.com/IntelRealSense/librealsense/issues/3682#issuecomment-642344385
dist_list = []
disp_list = []
num_matches_list = []
pointclouds = []
pointclouds_T = []
frame_error = []
R_list = []
t_list = []

sel_frames = [0] #for generalization
fig_number = 0
trans_number = 0

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
        if(get_depth):
            print(frame_number)
            if frame_number in sel_frames:
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

                radius = 0.15

                pointcloud_X = point_cloud_in_numpy[:,0]
                pointcloud_Y = point_cloud_in_numpy[:,1]
                pointcloud_Z = point_cloud_in_numpy[:,2]
                pointcloud_mask = 2*(pointcloud_X**2) + 2*(pointcloud_Y**2) + 2*(pointcloud_Z**2) <= 3*(radius**2)

                data = point_cloud_in_numpy[pointcloud_mask]
                path_save = pc_path + "/{:04d}.npy".format(frame_number)


                print('a')
                if(frame_number==0):
                    data_str = data
                    data0 = data
                    np.save(path_save,data_str)
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
                        P0,P1 = filter_invalid_3D_points(P0,P1,0.01)
                        if(frame_number==609):
                            D0 = P0
                            D1 = P1
                        A = np.mat(P0)
                        B = np.mat(P1)
                        n = A.shape[0]
                        # recover the transformation
                        Rc, tc = euclidean_transform_3D(A, B)
                        if(trans_number==0):
                            R_list.append(Rc)
                            t_list.append(tc)
                            trans_number+=1

                        else:
                            for i, (Ri, ti) in enumerate(zip(R_list, t_list)):
                                #print(f'Índice: {i}, Elemento da lista 1: {item1}, Elemento da lista 2: {item2}')
                                # Combine as rotações
                                R_combined = np.dot(Rc, Ri)
                                # Combine as translações
                                t_combined = np.dot(Rc, ti) + tc
                                R_list[i] = R_combined
                                t_list[i] = t_combined
                                trans_number += 1
                            R_list.append(Rc)
                            t_list.append(tc)
                        pc1 = np.array(data1)
                        np.save(path_save, pc1)
                        P0 = P0_next
            else:
                print("pass")
        else:
            print(frame_number, dist)
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
            if(frame_number==0):
                matches_list = []
                img_last = color_frame_npy
            else:
                img_now = color_frame_npy
                # if(pair_zero_flag):
                #     pair_zero_flag = False
                #     image1 = img_last
                #     image2 = img_now
                #     img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
                #     img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
                #     # Initiate SIFT detector
                #     sift = cv.SIFT_create()
                #     # find the keypoints and descriptors with SIFT
                #     kp1, des1 = sift.detectAndCompute(img1, None)
                #     kp2, des2 = sift.detectAndCompute(img2, None)
                # else:
                #     image2 = img_now
                #     img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
                #     kp2, des2 = sift.detectAndCompute(img2, None)
                #     img_next = img2

                image1 = img_last
                image2 = img_now
                img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
                img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
                # Initiate SIFT detector
                sift = cv.SIFT_create()
                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(img1, None)
                kp2, des2 = sift.detectAndCompute(img2, None)
                #img_next = img2
                #des_next = des2
                #kp_next = kp2
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                # store all the good matches as per Lowe's ratio test.
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                if len(good) > MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)  # find Homography just to use RANSAC isn't optimazed, but it`s implemented!
                    matchesMask = mask.ravel().tolist()
                    h, w = img1.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)
                    pts_cent = centeroidnp(pts)
                    dst_cent = centeroidnp(dst)
                    dist_var = np.linalg.norm(dst_cent - pts_cent)
                    img2_l = img2
                    img2_l = cv.polylines(img2_l, [np.int32(dst)], True, 200, 3, cv.LINE_AA)
                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                       singlePointColor=None,
                                       matchesMask=matchesMask,  # draw only inliers
                                       flags=2)
                    img3 = cv.drawMatches(img1, kp1, img2_l, kp2, good, None, **draw_params)
                    matchesMask_np = np.asarray(matchesMask)
                    src_pts_mask = src_pts[matchesMask_np == 1]
                    dst_pts_mask = dst_pts[matchesMask_np == 1]
                    num_mat = len(src_pts_mask)
                    valid_matches = np.stack((src_pts_mask[:, 0], dst_pts_mask[:, 0]), axis=0)
                    dist_var = np.mean(reject_outliers(np.diag(distance.cdist(valid_matches[0], valid_matches[1]))))
                    disp_var = variation(np.diag(distance.cdist(valid_matches[0], valid_matches[1])))
                    # return valid_matches, dist_var, disp_var, num_mat
                    matches = valid_matches
                    dist = dist_var
                    disp = disp_var
                    num_matches = num_mat
                    if (frame_number == 1):
                        # video
                        height, width, layers = img3.shape
                        fourcc = cv.VideoWriter_fourcc(*'mp4v')
                        video = cv.VideoWriter(video_name, fourcc, fps, (width, height))
                    if(num_matches<MAX_MATCH_COUNT):
                        if(dist>MIN_DIST):
                            video.write(img3)
                            path_save_img = img_path + "/{:04d}".format(frame_number)
                            #imgkp1 = cv.drawKeypoints(img1, kp1, image1)
                            #imgkp2 = cv.drawKeypoints(img2, kp2, image2)
                            cv.imwrite(path_save_img + "-1a.jpg", image1)
                            cv.imwrite(path_save_img + "-2a.jpg", image2)
                            cv.imwrite(path_save_img + "-1b.jpg", img1)
                            cv.imwrite(path_save_img + "-2b.jpg", img2)
                            cv.imwrite(path_save_img + "-3.jpg",img3)
                            sel_frames.append(frame_number)
                            matches_list.append(matches)
                            num_matches_list.append(num_matches)
                            dist_list.append(dist)
                            disp_list.append(disp)
                            img_last = color_frame_npy
                            #img1 = img_next
                            #kp1= kp_next
                            #des1= des_next
                else:
                    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                    matchesMask = None
                    frame_error.append(frame_number)
        frame_number += 1  # get number of frames
        playback.resume()
    # Cleanup:
    cv.destroyAllWindows()
    video.release()
    pipe.stop()
    print("Video Genereted")
img_end = color_frame_npy
pc_list = sorted(os.listdir(pc_path))
print("0 ",pc_list)
pc_list = pc_list[:-2]
print("1 ",pc_list)
plot3D(0,pc_path)
Rlen = len(R_list)


for i, npc in enumerate(pc_list):
    print(i)
    load_path = pc_path + "/" + npc
    p = np.load(load_path)
    A = np.mat(p)
    n = A.shape[0]
    if (i < Rlen):
        A_transformed = (R_list[i] * A.T) + np.tile(t_list[i], (1, n))
        A_transformed = A_transformed.T
        pc1 = np.array(A_transformed)
        load_path = trans_path + "/T" + npc
        np.save(load_path, pc1)
plot3D(1,pc_path)

# pontos de referência sem trans
x0 = D0[:,0]
y0 = D0[:,1]
z0 = D0[:,2]

x1 = D1[:,0]
y1 = D1[:,1]
z1 = D1[:,2]
# Create a 3D figure
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='red')
ax.scatter(x1,y1,z1,s=10.0,color='blue')
# Initialize lists to collect all point coordinates
all_x, all_y, all_z = [], [], []

# Collect the points for later adjustment of the aspect ratio
all_x.extend(x0)
all_y.extend(y0)
all_z.extend(z0)
all_x.extend(x1)
all_y.extend(y1)
all_z.extend(z1)

# Calculate the ranges for each axis
max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
mid_x = (max(all_x) + min(all_x)) * 0.5
mid_y = (max(all_y) + min(all_y)) * 0.5
mid_z = (max(all_z) + min(all_z)) * 0.5

# Set the limits for each axis to make them equal
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# pontos de referência com trans

A = np.mat(D0)
B = np.mat(D1)
n = A.shape[0]
# recover the transformation
Rc, tc = euclidean_transform_3D(A, B)

A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T
D0T = np.asarray(A_transformed)
D1T = np.asarray(D1)

x0 = D0T[:,0]
y0 = D0T[:,1]
z0 = D0T[:,2]

x1 = D1T[:,0]
y1 = D1T[:,1]
z1 = D1T[:,2]

# Create a 3D figure
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='red')
ax.scatter(x1,y1,z1,s=10.0,color='blue')
# Initialize lists to collect all point coordinates
all_x, all_y, all_z = [], [], []

# Collect the points for later adjustment of the aspect ratio
all_x.extend(x0)
all_y.extend(y0)
all_z.extend(z0)
all_x.extend(x1)
all_y.extend(y1)
all_z.extend(z1)

# Calculate the ranges for each axis
max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
mid_x = (max(all_x) + min(all_x)) * 0.5
mid_y = (max(all_y) + min(all_y)) * 0.5
mid_z = (max(all_z) + min(all_z)) * 0.5

# Set the limits for each axis to make them equal
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
ax.set_title('After TRANS')

# nuvens de pontos com trans

data_str = np.load(pc_path + "/0453.npy")
data_end = np.load(pc_path + "/0609.npy")

x0 = data_str[::1000, 0]
y0 = data_str[::1000, 1]
z0 = data_str[::1000, 2]

x1 = data_end[::1000, 0]
y1 = data_end[::1000, 1]
z1 = data_end[::1000, 2]

# Create a 3D figure
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='blue')
ax.scatter(x1,y1,z1,s=10.0,color='red')

# Initialize lists to collect all point coordinates
all_x, all_y, all_z = [], [], []

# Collect the points for later adjustment of the aspect ratio
all_x.extend(x0)
all_y.extend(y0)
all_z.extend(z0)
all_x.extend(x1)
all_y.extend(y1)
all_z.extend(z1)

# Calculate the ranges for each axis
max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
mid_x = (max(all_x) + min(all_x)) * 0.5
mid_y = (max(all_y) + min(all_y)) * 0.5
mid_z = (max(all_z) + min(all_z)) * 0.5

# Set the limits for each axis to make them equal
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_title('Nuvens de pontos antes da transformada de coordenadas')

# pontos de referência sem trans

A = np.mat(data_str)
B = np.mat(data_end)
n = A.shape[0]
A_transformed = (Rc*A.T) + np.tile(tc, (1, n))
A_transformed = A_transformed.T
data_str_T = np.asarray(A_transformed)
data_end_T = np.asarray(data_end)

x0 = data_str_T[::1000, 0]
y0 = data_str_T[::1000, 1]
z0 = data_str_T[::1000, 2]

x1 = data_end_T[::1000, 0]
y1 = data_end_T[::1000, 1]
z1 = data_end_T[::1000, 2]

# Create a 3D figure
fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')

#Plot the point cloud data
ax.scatter(x0,y0,z0,s=10.0,color='blue')
ax.scatter(x1,y1,z1,s=10.0,color='red')

# Initialize lists to collect all point coordinates
all_x, all_y, all_z = [], [], []

# Collect the points for later adjustment of the aspect ratio
all_x.extend(x0)
all_y.extend(y0)
all_z.extend(z0)
all_x.extend(x1)
all_y.extend(y1)
all_z.extend(z1)

# Calculate the ranges for each axis
max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
mid_x = (max(all_x) + min(all_x)) * 0.5
mid_y = (max(all_y) + min(all_y)) * 0.5
mid_z = (max(all_z) + min(all_z)) * 0.5

# Set the limits for each axis to make them equal
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Nuvens de pontos após da transformada de coordenadas')
