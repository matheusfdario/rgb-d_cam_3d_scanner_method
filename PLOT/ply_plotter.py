import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import open3d as o3d

import matplotlib as mpl
mpl.use("Qt5Agg")

# Read .ply file
input_file = r"C:\Users\Victor\Documents\realsense\ply\_1710512713820.04589843750000.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

# Visualize the point cloud within open3d
o3d.visualization.draw_geometries([pcd])

point_cloud_in_numpy = np.asarray(pcd.points)

data = point_cloud_in_numpy

# Split the data into x, y, and z arrays
x = data[::100, 0]
y = data[::100, 1]
z = data[::100, 2]

# Create a 3D figure
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(111, projection='3d')

# Plot the point cloud data
# ax.scatter(x, y, z, s=0.01)
ax.plot_trisurf(x, y, z, cmap='jet')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

# Show the plot
plt.show()

# # o3d.io.write_image("filename.png",pcd.Image)

# # Calcular a profundidade máxima e mínima para normalização
# z_min = np.min(z)
# z_max = np.max(z)
#
# # Normalizar as profundidades para o intervalo [0, 1]
# normalized_z = (z - z_min) / (z_max - z_min)
#
# # Criar uma nova figura
# fig = plt.figure()
#
# # Criar um subplot 2D
# ax = fig.add_subplot(111)
#
# # Plotar os pontos como uma nuvem de pontos 2D, com a cor definida pela profundidade
# ax.scatter(x, y, c=normalized_z, cmap='viridis')
#
# # Definir rótulos do eixo x e y
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
#
# # Mostrar a plotagem
# plt.show()

