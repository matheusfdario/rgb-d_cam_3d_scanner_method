import open3d as o3d
import numpy as np
from matplotlib import cm

# Load the .obj file
mesh = o3d.io.read_triangle_mesh("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MESH/hex-03.obj")

# Check if the mesh is loaded correctly
if mesh.is_empty():
    print("Failed to load mesh.")
else:
    print("Mesh loaded successfully.")

    # Get vertex positions
    vertices = np.asarray(mesh.vertices)

    # Calculate the centroid
    centroid = np.mean(vertices, axis=0)

    # Calculate distances from each vertex to the centroid
    distances = np.linalg.norm(vertices - centroid, axis=1)

    # Normalize the distances to [0, 1]
    normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Apply a colormap (e.g., viridis)
    colormap = cm.get_cmap('nipy_spectral')
    colors = colormap(normalized_distances)[:, :3]  # Get RGB viridisvalues

    # Assign colors to the mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
# Calculate the normals of the vertex
mesh.compute_vertex_normals()# Paint it gray. Not necessary but the reflection of lighting is hardly perceivable with black surfaces.
color = (np.array([[129],[81],[56]]))/255
#mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
#mesh.paint_uniform_color(color)# Visualize the mesh


o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True,point_show_normal=True,mesh_show_wireframe=False)

# import open3d as o3d
# import numpy as np
# from matplotlib import cm
#
# # Função para calcular a distância de um ponto para a linha definida por dois pontos
# def point_to_line_distance(point, p1, p2):
#     # Vetor da linha
#     line_vector = p2 - p1
#     # Vetor do ponto para o p1
#     point_vector = point - p1
#     # Projeção do ponto sobre a linha
#     line_unit_vector = line_vector / np.linalg.norm(line_vector)
#     projection = np.dot(point_vector, line_unit_vector)
#     projection_vector = projection * line_unit_vector
#     # Distância entre o ponto e a linha
#     distance = np.linalg.norm(point_vector - projection_vector)
#     return distance
#
# # Load the .obj file
# mesh = o3d.io.read_triangle_mesh("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MESH/mesh-dani.obj")
#
# # Check if the mesh is loaded correctly
# if mesh.is_empty():
#     print("Failed to load mesh.")
# else:
#     print("Mesh loaded successfully.")
#
#     # Get vertex positions
#     vertices = np.asarray(mesh.vertices)
#
#     # Define the two points manually (replace with your points)
#     p1 = np.array([0.0, 0.0, 0.0])  # Point 1
#     p2 = np.array([0.1, 0.1, 0.0])  # Point 2
#
#     # Calcular as distâncias de cada vértice para a linha definida por p1 e p2
#     distances_to_line = np.array([point_to_line_distance(v, p1, p2) for v in vertices])
#
#     # Determinar o menor e maior raio baseados nas distâncias dos pontos da malha
#     r_min = distances_to_line.min()  # Raio mínimo: menor distância
#     r_max = distances_to_line.max()  # Raio máximo: maior distância
#
#     print(f"Raio mínimo: {r_min}")
#     print(f"Raio máximo: {r_max}")
#
#     # Normalizar as distâncias dentro do intervalo [0, 1]
#     normalized_distances = np.clip((distances_to_line - r_min) / (r_max - r_min), 0, 1)
#
#     # Aplicar o colormap (ex: 'flag' ou outro de sua escolha)
#     colormap = cm.get_cmap('viridis')  # Você pode alterar para 'plasma', 'inferno', etc.
#     colors = colormap(normalized_distances)[:, :3]  # Obter valores RGB
#
#     #
