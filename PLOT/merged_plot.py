from matplotlib import pyplot as plt
import numpy as np  # fundamental package for scientific computing
filepath = '/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MERGED/out.npy'
merged_pointcloud =  np.load(filepath)
# plot transformed merged pointcloud
x = merged_pointcloud[::1000, 0]
y = merged_pointcloud[::1000, 1]
z = merged_pointcloud[::1000, 2]
# Create a 3D figure
fig = plt.figure(0)

ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
ax.scatter(x, y, z,s=0.1, color='green')
# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')
ax.set_title('Merged Pointcloud')
plt.show()