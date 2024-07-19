import numpy as np
import open3d as o3d
import os

#var
dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-07-2024_14-44-02"
pc_list = sorted(os.listdir(dir_path))


for i,npc in enumerate(pc_list):
    load_path = dir_path + "/" + npc
    print(load_path)
    pc = np.load(load_path)
    if(i==0):
        xyz = pc
    else:
        xyz = np.vstack((xyz, pc))
    np.save(load_path,pc)


# Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MERGED/data.ply", pcd)

o3d.visualization.draw_geometries([pcd])