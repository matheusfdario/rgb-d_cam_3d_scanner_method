import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

#var
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-07-2024_14-44-02"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/12-08-2024_18-16-36"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/20-08-2024_10-54-02"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/14-11-2024_16-44-25"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_10-28-03"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_13-28-40" #ok
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_14-01-56"
#dir_path =  "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/06-12-2024_11-35-37"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/07-11-2024_21-25-44"

dir_path =  "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/11-12-2024_15-14-19"
pc_list = sorted(os.listdir(dir_path))

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def plot3D(f_num, path):
    pc_list = sorted(os.listdir(path))

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
        x = pc[::10000, 0]
        y = pc[::10000, 1]
        z = pc[::10000, 2]

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

plot3D(1,dir_path)

# # Create a 3D figure
# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
#
# # Set the axis labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('equal')
# ax.set_title('Pointclouds')
#
#
#
# colors = cm.rainbow(np.linspace(0, 1, len(pc_list)))
# for i,npc in enumerate(pc_list):
#
#     load_path = dir_path + "/" + npc
#     print(load_path)
#     pc = np.load(load_path)
#     x = pc[::10000,0]
#     y = pc[::10000,1]
#     z = pc[::10000,2]
#
#     #Plot the point cloud data
#     ax.scatter(x,y,z,s=1.0,color=colors[i],alpha=0.75)
#     #np.save(load_path,pc)