import numpy as np
from open3d import * 
import pickle as pickle
# import open3d_tutorial as o3dtut



def main():    
    with open(f'pcl_cam_pcl2.pickle','rb') as input_pcl:
        pcs = pickle.load(input_pcl)
    print(pcs.shape)
    for view in range(len(pcs)):
        pcd = geometry.PointCloud()
        cur_pc = pcs[view]
        print(f"shape of cur pc {cur_pc.shape}")
        print(f"mean of {view} is {np.mean(cur_pc,axis=0)}")
        cur_pc = cur_pc.reshape(-1,3)
        pcd.points = utility.Vector3dVector(cur_pc)
        visualization.draw_geometries_with_custom_animation([pcd]) # Visualize the point cloud


if __name__ == "__main__":
	main()