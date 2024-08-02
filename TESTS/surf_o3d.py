import open3d as o3d
import numpy as np
import matplotlib.cm as plt
import matplotlib.cm
# Loading and visualizing a PLY point cloud
print("Loading a PLY point cloud, printing, and rendering...")
file_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MERGED/data.ply"
out_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MESH/mesh.obj"
pcd = o3d.io.read_point_cloud(file_path)
# Printing point cloud information and points array
print(pcd)
print(np.asarray(pcd.points))
# Setting visualization parameters
# view_params = {
#     "zoom": 0.3412,
#     "front": [0.4257, -0.2125, -0.8795],
#     "lookat": [2.6172, 2.0475, 1.532],
#     "up": [-0.0694, -0.9768, 0.2024]
# }
# Rendering the point cloud
#o3d.visualization.draw_geometries([pcd], **view_params)
#o3d.visualization.draw_geometries([pcd])
print("Downsampling the point cloud with a voxel size of 0.001")
# Applying voxel downsampling
downpcd = pcd.voxel_down_sample(voxel_size=0.0005)
# # Setting visualization parameters for the downsampled point cloud
# downsample_view_params = {
#     "zoom": 0.3412,
#     "front": [0.4257, -0.2125, -0.8795],
#     "lookat": [2.6172, 2.0475, 1.532],
#     "up": [-0.0694, -0.9768, 0.2024]
# }
# Rendering the downsampled point cloud
# o3d.visualization.draw_geometries([downpcd], **downsample_view_params)
#o3d.visualization.draw_geometries([downpcd])

print("Recomputing normals for the downsampled point cloud")
# Estimating normals for the downsampled point cloud
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# Visualization parameters for displaying normals
# normals_view_params = {
#     "zoom": 0.3412,
#     "front": [0.4257, -0.2125, -0.8795],
#     "lookat": [2.6172, 2.0475, 1.532],
#     "up": [-0.0694, -0.9768, 0.2024],
#     "point_show_normal": True
# }
# Rendering the downsampled point cloud with normals
#o3d.visualization.draw_geometries([downpcd])

# bunny = o3d.data.BunnyMesh()
# gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
#
# pcd = gt_mesh.sample_points_poisson_disk(5000)
# pcd.normals = o3d.utility.Vector3dVector(np.zeros(
#     (1, 3)))  # invalidate existing normals
#
pcd = downpcd
pcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#
#
#

print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
print(mesh)
# o3d.visualization.draw_geometries([mesh])

print('visualize densities')
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh])


print('remove low density vertices')
vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)
print(mesh)
# Calculate the normals of the vertex
mesh.compute_vertex_normals()
# Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
color = (np.array([[129],[81],[56]]))/255
#mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
mesh.paint_uniform_color(color)

o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
#o3d.io.write_triangle_mesh(out_path, mesh, compressed=True, print_progress=True)

