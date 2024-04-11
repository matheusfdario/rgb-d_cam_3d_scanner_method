import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import open3d as o3d

# Read .ply file
#input_file = "out.ply"
input_file = "ply_1710437557553.69335937500000.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
o3d.visualization.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format.
point_cloud_in_numpy = np.asarray(pcd.points)

# Split the data into x, y, and z arrays
x = point_cloud_in_numpy[:, 0]
y = point_cloud_in_numpy[:, 1]
z = point_cloud_in_numpy[:, 2]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(x, y, z, s=1)

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

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