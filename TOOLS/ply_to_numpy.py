import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import open3d as o3d

# Read .ply file
#input_file = "out.ply"
input_file = "/home/matheusfdario/Documentos/LASSIP/rep/pyrealsense_test/alveolo.ply"
outfile = "alveolo.npy"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
#o3d.visualization.draw_geometries([pcd])

# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format.
point_cloud_in_numpy = np.asarray(pcd.points)
np.save(outfile, point_cloud_in_numpy)