import numpy as np
import open3d as o3d
import os

#var
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-07-2024_14-44-02"
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/14-11-2024_16-44-25" # real big
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_12-56-58" # dani pipe
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_13-28-40" # dani pipe 2
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_14-01-56" # spool slice a
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_14-55-00" # spool slice a 2
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/19-11-2024_16-32-40" # spool slice b 1
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/MESH-DIR/tati_pipe_19-11-2024_17-19-31" # tati pipe
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/06-12-2024_12-53-13" # hex 1
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/10-12-2024_15-08-54" # hex 2
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/16-12-2024_18-57-41" #meia cana
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/17-12-2024_11-52-22" # test spray 3d
#dir_path = "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/17-12-2024_16-27-15" # teste shampoo seco
dir_path =  "/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/PC-T/29-01-2025_19-13-47" # hex 3
pc_list = sorted(os.listdir(dir_path))
buffer_max_size = 40
buffer_size = 0
xyz_create_flag = True

for i,npc in enumerate(pc_list):
    load_path = dir_path + "/" + npc
    #print(load_path)
    pc = np.load(load_path)
    if(i==0):
        xyz_buffer = pc
        buffer_size +=1
        #xyz = pc
    else:
        print(i,buffer_size,i%buffer_max_size)
        if(i%buffer_max_size==0):
            buffer_size = 0
            if(xyz_create_flag):
                xyz = xyz_buffer
                xyz_create_flag = False
            else:
                xyz = np.vstack((xyz, xyz_buffer))
        else:
            if(buffer_size==0):
                xyz_buffer = pc
            else:
                xyz_buffer = np.vstack((xyz_buffer, pc))
            buffer_size += 1
        #xyz = np.vstack((xyz, pc))
    #np.save(load_path,pc)
if(buffer_size>0):
    xyz = np.vstack((xyz, xyz_buffer))

# Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("/media/matheusfdario/HD/REPOS/rgb-d_cam_3d_scanner_method/DATA/MERGED/data.ply", pcd)

o3d.visualization.draw_geometries([pcd])