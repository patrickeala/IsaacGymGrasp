import pickle
import numpy as np
from numpy.core.numeric import argwhere
import utils
import time
from IsaacGymSimulator2 import IsaacGymSim
from isaacgym import gymutil
from isaacgym.torch_utils import *
import gc    
import torch
from pathlib import Path
import os

# execute positive candidates
# device = torch.device('cuda:0')



if __name__ == "__main__":

    # object = utils.args.object

    object_to_code = {
        "sugar_box"      :"004_sugar_box",
        "bowl"           :"024_bowl",
        "tomato_soup_can":'005_tomato_soup_can', 
        "potted_meat_can":'010_potted_meat_can', 
        "mug"            :'025_mug', 
        "foam_brick"     : '061_foam_brick', 
        "j_cups"         :'065_j_cups',
        "sponge"         :'026_sponge'
        }
    # serial_number = utils.args.serial_number #"1641738054"
    # threshold = utils.args.threshold
    # initialize global variables
    # data_path = f'experiment2/{threshold}/{object_to_code[object]}'
    # save_path = f'experiment2/{threshold}/{object_to_code[object]}'
    cat = "cylinder"
    idx = 11
    data_path = f'experiment4/{cat}/{cat}{idx:03}/'
    save_path = data_path
    path_to_assets = '../grasper/grasp_data/' # object mesh and json path
    scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{cat}{idx:03}.json')
    headless = False

        
    obj_name = object
    if cat in ["box","cylinder"]:
        path_to_obj_mesh = f'/home/user/grasper/grasp_data/meshes/{cat}/{cat}{idx:03}.stl'
    else:
        path_to_obj_mesh = f'/home/user/grasper/grasp_data/meshes/{cat}/{cat}{idx:03}.obj'
    # if os.path.isfile(f'{save_path}/final_grasps_labelled.npz'):
    #     print("already lebelled")
    # else:

    # # get initial run
    # graspnet_initial_grasps_fname = f'{data_path}/grasps_graspnet_initial.npz'
    # distances = pickle.load(open(f"{data_path}/grasps_graspnet_initial_distances.pkl","rb"))
    # quaternions, translations, obj_pose_relative = utils.load_experiment_data(graspnet_initial_grasps_fname)
    # path_to_assets = '../grasper/grasp_data/' # object mesh and json path
    # scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    # print(f"getting labels for {obj_name}")
    # print(f"relative obj position is {obj_pose_relative}")
    # isaac_labels = utils.simulate_isaac(quaternions, translations, obj_pose_relative, scale=scale,
    #             path_to_obj_mesh=path_to_obj_mesh, headless=headless,grasps_distances=distances)

    # np.savez_compressed(f'{save_path}/grasps_graspnet_initial_labelled.npz',
    #                     quaternions=quaternions,
    #                     translations=translations,
    #                     isaac_labels=isaac_labels)
    # del quaternions, translations, isaac_labels



    # graspnet_final_grasps_fname = f'{data_path}/grasps_graspnet_final.npz'
    # distances = pickle.load(open(f"{data_path}/grasps_graspnet_final_distances.pkl","rb"))
    # # print("distance is : ",distances)

    # quaternions, translations, obj_pose_relative = utils.load_experiment_data(graspnet_final_grasps_fname)
    # print(f"getting labels for {obj_name}")
    # print(f"relative obj position is {obj_pose_relative}")
    # isaac_labels = utils.simulate_isaac(quaternions, translations, obj_pose_relative, scale=scale,
    #             path_to_obj_mesh=path_to_obj_mesh, headless=headless,grasps_distances=distances)

    # np.savez_compressed(f'{save_path}/grasps_graspnet_final_labelled.npz',
    #                     quaternions=quaternions,
    #                     translations=translations,
    #                     isaac_labels=isaac_labels)
    # del quaternions, translations, isaac_labels
    



    heuristic_initial_grasps_fname = f'{data_path}/grasps_heuristics_initial.npz'
    # distances = pickle.load(open(f"{data_path}/grasps_heuristics_initial_distances.pkl","rb"))
    distances = None

    quaternions, translations, obj_pose_relative = utils.load_experiment_data(heuristic_initial_grasps_fname)
    
    print(f"getting labels for {obj_name}")
    print(f"relative obj position is {obj_pose_relative}")
    isaac_labels = utils.simulate_isaac(quaternions, translations, obj_pose_relative, scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless)

    np.savez_compressed(f'{save_path}/grasps_heuristics_initial_labelled.npz',
                        quaternions=quaternions,
                        translations=translations,
                        isaac_labels=isaac_labels)
    del quaternions, translations, isaac_labels



    heuristic_final_grasps_fname = f'{data_path}/grasps_heuristics_final.npz'
    # distances = pickle.load(open(f"{data_path}/grasps_heuristics_final_distances.pkl","rb"))
    distances = None
    quaternions, translations, obj_pose_relative = utils.load_experiment_data(heuristic_final_grasps_fname)
    print(f"getting labels for {obj_name}")
    print(f"relative obj position is {obj_pose_relative}")
    isaac_labels = utils.simulate_isaac(quaternions, translations, obj_pose_relative, scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless)

    np.savez_compressed(f'{save_path}/grasps_heuristics_final_labelled.npz',
                        quaternions=quaternions,
                        translations=translations,
                        isaac_labels=isaac_labels)
    del quaternions, translations, isaac_labels

        # final_grasps_fname = f'{data_path}/final_grasps.npz'
        # # idx = np.array([504, 747, 237 ,202, 548, 248,  31, 758, 752, 802])
        
        # quaternions, translations, obj_pose_relative = utils.load_experiment_data(final_grasps_fname)

        # # quaternions = quaternions[idx]
        # # translations = translations[idx]

        # isaac_labels = utils.simulate_isaac(quaternions, translations, obj_pose_relative, scale=1,
        #             path_to_obj_mesh=path_to_obj_mesh, headless=headless)




        # np.savez_compressed(f'{save_path}/final_grasps_labelled.npz',
        #                     quaternions=quaternions,
        #                     translations=translations,
        #                     isaac_labels=isaac_labels)
        # del quaternions, translations, isaac_labels
