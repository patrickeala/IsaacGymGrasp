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
import sys

# execute positive candidates
# device = torch.device('cuda:0')



if __name__ == "__main__":

    
    # initialize global variables
    data_path = 'grasp_data_generated'
    save_path = 'grasp_data_generated'
    path_to_assets = '../grasper/grasp_data/' # object mesh and json path

    MAX_ENVS = 100000
    headless = False
    trial = 100
    cat = "mug"#'mug'
    idx = 1
    utils.args.settings = 1
    obj_name = f'{cat}{idx:03}'
    

    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
    if utils.args.settings == 1:
        path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
    scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    

    # raw_data_name = f'{data_path}/{cat}/{obj_name}/main{trial}.npz'
    # _ , _, obj_pose_relative, _ = utils.load_grasp_data(raw_data_name)
    labelled_data_name = f'/home/gpupc2/GRASP/grasper/semantic_dataset/mug/mug001/all_stable_grasps.npz'
    quaternions, translations = np.load(labelled_data_name)
    # indices = np.argwhere(isaac_labels == 1)
    # indices = indices.squeeze()

    # if len(indices) == 0:
    #     exit(1)
    # print(f"getting labels for {obj_name}")

    isaac_labels = np.zeros(len(translations))
    isaac_labels_promising = utils.simulate_isaac(quaternions[0], translations[0], [0,0,0], scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless)


    
    # for part_indices in splitted_indices:
    #     isaac_labels_promising = simulate_isaac(quaternions[part_indices], translations[part_indices], obj_pose_relative, scale=scale,
    #                 path_to_obj_mesh=path_to_obj_mesh, headless=headless, settings=settings)
    #     isaac_labels[part_indices] = isaac_labels_promising

    #     print(f'Simulated: {len(part_indices)}')
    #     print(f'Positive numbers {np.sum(isaac_labels_promising)}')
    #     # print("sleeping ...........")
    #     # time.sleep(3)
    # np.savez_compressed(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}_rest.npz',
    #                 quaternions=quaternions,
    #                 translations=translations,
    #                 isaac_labels = isaac_labels)


