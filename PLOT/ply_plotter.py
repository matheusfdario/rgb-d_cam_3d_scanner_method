import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import open3d as o3d

# Read .ply file
#input_file = "out.ply"
input_file = "/home/matheusfdario/Documentos/LASSIP/rep/pyrealsense_test/fenda.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
o3d.visualization.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format.
point_cloud_in_numpy = np.asarray(pcd.points)

# max_size = point_cloud_in_numpy.shape[0]
# view_size = 20000
# limit_inf = int(max_size/2-view_size/2)
# limit_sup = int(max_size/2+view_size/2)

#data = point_cloud_in_numpy[limit_inf:limit_sup]


point_cloud_valid_only = np.zeros_like(point_cloud_in_numpy)

z_min = -0.2

cont = 0

for i in range(point_cloud_in_numpy.shape[0]):
    if(point_cloud_in_numpy[i,2]>z_min):
        point_cloud_valid_only[cont] = point_cloud_in_numpy[i]
        cont = cont + 1

data = np.zeros([cont+1,3])
data = point_cloud_valid_only[0:cont]

# Split the data into x, y, and z arrays
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(x, y, z, s=1)

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
# Show the plot
plt.show()



#
#
#
# def get_pts(infile):
#     data = np.loadtxt(infile, delimiter=',')
#     return data[12:, 0], data[12:, 1], data[12:, 2]  # returns X,Y,Z points skipping the first 12 lines
#
#
# def plot_ply(infile):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     x, y, z = get_pts(infile)
#     ax.scatter(x, y, z, c='r', marker='o')
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.show()
#
#
# if __name__ == '__main__':
#     infile = 'out.ply'
#     plot_ply(infile)