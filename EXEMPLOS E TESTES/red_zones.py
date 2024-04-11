import open3d as o3d
import numpy as np
import cv2

# Carregar a imagem RGB original
rgb_image_path = "/media/matheusfdario/HD/REALSENSE/test/PNG/png_Color_1710437557553.99731445312500.png"
rgb_image = cv2.imread(rgb_image_path)

# Carregar o arquivo .ply
ply_path = "/media/matheusfdario/HD/REALSENSE/test/PLY/ply_1710437557553.69335937500000.ply"
pcd = o3d.io.read_point_cloud(ply_path)

limiar = -1.8

# Converter a nuvem de pontos para um array numpy
points1 = np.asarray(pcd.points)
x_axis = np.sort(points1[:,0])
y_axis = np.sort(points1[:,1])
points = points1[points1[:,2]>limiar]
x_max = np.max(points[:,0])
y_max = np.max(points[:,1])
x_min = np.min(points[:,0])
y_min = np.min(points[:,1])

# Determinar a resolução da imagem RGB
height, width, _ = rgb_image.shape
#height = 480
#width = 848
#x_axis = np.linspace(x_min,x_max,height)
#y_axis = np.linspace(y_min,y_max,width)

#x_axis = np.sort(points[:,0])
#y_axis = np.sort(points[:,1])



# Criar uma nova imagem colorida com base na resolução da imagem RGB
new_image = np.zeros((height, width, 3), dtype=np.uint8)

# Mapear os pontos na nuvem de pontos para as coordenadas de pixels na nova imagem
for point in points:
    x_pixel = (np.abs(x_axis - point[0])).argmin()
    y_pixel = (np.abs(y_axis - point[1])).argmin()
    #x = int(point[0] * (width - 1))  # Escalar para a resolução da imagem RGB
    #y = int(point[1] * (height - 1))
    #if 0 <= x < width and 0 <= y < height:  # Verificar se o ponto está dentro dos limites da imagem
    #    new_image[y, x] = rgb_image[y, x]  # Atribuir a cor correspondente da imagem RGB
    new_image[x_pixel, y_pixel] = rgb_image[x_pixel, y_pixel]  # Atribuir a cor correspondente da imagem RGB
'''
# Mapear os pontos na nuvem de pontos para as coordenadas de pixels na nova imagem
for point in points:
    x = int(point[0] * width)  # Escalar para a resolução da imagem RGB
    y = int(point[1] * height)
    if 0 <= x < width and 0 <= y < height:  # Verificar se o ponto está dentro dos limites da imagem
        new_image[y, x] = rgb_image[y, x]  # Atribuir a cor correspondente da imagem RGB
'''
# Especificar uma cor arbitrária para os pixels ausentes
arbitrary_color = (255, 255, 255)  # Branco
new_image[np.where((new_image == [0, 0, 0]).all(axis=2))] = arbitrary_color

# Salvar a nova imagem colorida
new_image_path = "/media/matheusfdario/HD/REALSENSE/test/img/imagem_rgb_3.jpg"
cv2.imwrite(new_image_path, new_image)

print("Nova imagem colorida gerada e salva com sucesso!")
