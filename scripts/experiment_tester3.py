import pickle5 as pickle
import numpy as np
from numpy.core.numeric import argwhere
from sklearn.covariance import graphical_lasso
import utils
import time
from IsaacGymSimulator2 import IsaacGymSim
from isaacgym import gymutil
from isaacgym.torch_utils import *
import gc    
import torch
from pathlib import Path
import sys

# execute positive candidates
# device = torch.device('cuda:0')



if __name__ == "__main__":

    object = "sugar_box"
    headless = False
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


    # initialize global variables
    info, args = pickle.load(open('experiments/results/experiment1/004_sugar_box/1641738054/info', 'rb'))
    grasps_distances = np.asarray(pickle.load(open('gripper_to_obj_distances.pkl', 'rb')))
    print(info.keys())
    print(grasps_distances)
    success_last = info['success'][-1, :]
    
    # [504 747 237 202 548 248  31 758 752 802 407 875 725 285 341 831 801 938 265 
    # 406 882 993 931 239 588 818 523 673 471 631 869 812 724 309 554 150 525 218 
    # 23 726 193 570 979 362 549 872 748 947 773 255]
    # mask = np.argsort(success_last)[::-1][:N]
    # print(mask)

    # correct success labels : 237 752 931 869 812
    # nonstable but success : 202
    # collision : 31 875 725 341 406 882 724 150 726 549 
    # collision (sensitive) : 802 407 875 725 993 239 554 218 570 748 947
    # far : 504 747 548 248 758 (285) 831 (265) 588 818 (523) 471 525 193 979 773 255
    # abnormal : 801 938 673 631 309 23 362 872
    # mask = [588] 
    # mask = [801, 938, 673, 631, 309, 23, 362, 872]
    mask = np.argsort(success_last)[::-1][:50]
    print(mask)
    # target_mask = [802, 407, 875, 725, 993, 239, 554, 218, 570, 748, 947]
    # target_mask = [31, 875, 725, 341, 406, 882, 724, 150, 726, 549 ]
    # distance_mask = []
    # for m in target_mask:
    #     distance_mask.append(np.where(mask==m)[0][0])
    # distance_mask = np.asarray(distance_mask)
    # grasps_distances = grasps_distances[distance_mask]
    # mask = target_mask
    # print(distance_mask)
    # mask = [993]
    # mask = [mask[-1]]
    # grasps_distances = [grasps_distances[-1]]

    translations_init = info['translations'][0, mask, :]
    quaternions_init = info['quaternions'][0, mask, :]
    translations_final = info['translations'][-1, mask, :]
    quaternions_final = info['quaternions'][-1, mask, :]

    path_to_obj_mesh = f'assets/meshes_10_objects/{object_to_code[object]}/google_16k/textured.obj'
    
    # get initial run
    _, obj_pose_relative = pickle.load(open(f'experiments/data/experiment1/004_sugar_box.pkl','rb'))

    print(f"relative obj position is {obj_pose_relative}")
    isaac_labels = utils.simulate_isaac(quaternions_final, translations_final, obj_pose_relative, scale=1,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless,grasps_distances=grasps_distances)
    isaac_labels = isaac_labels.astype(int)
    print("index of successful grapsp: ",mask*isaac_labels )
    # np.savez_compressed(f'{save_path}/initial_grasps_labelled.npz',
    #                     quaternions=quaternions,
    #                     translations=translations,
    #                     isaac_labels=isaac_labels)
    # del quaternions, translations, isaac_labels

        
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
