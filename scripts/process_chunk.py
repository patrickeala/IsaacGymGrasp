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

    data_path = 'grasp_data_generated'
    save_path = 'grasp_data_generated'
    path_to_assets = '../grasper/grasp_data/' # object mesh and json path
    
    headless = True
    trial = utils.args.trial
    cat = utils.args.cat #'mug'
    idx = utils.args.idx
        
    obj_name = f'{cat}{idx:03}'
    

    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
    if utils.args.settings == 1:
        path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
    scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    
  


    with open(f'{data_path}/{cat}/{obj_name}_isaac/all_info_{trial}.pkl', 'rb') as f:
        all_info = pickle.load(f)
    [quaternions,translations,promising_indices, splitted_indices,isaac_labels,obj_pose_relative,current_chunk] = all_info
    
    print(current_chunk)
    part_indices = splitted_indices[current_chunk]
    
    isaac_labels_promising = utils.simulate_isaac(quaternions[part_indices], translations[part_indices], obj_pose_relative, scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless)
    isaac_labels[part_indices] = isaac_labels_promising

    print(f'Simulated: {len(part_indices)}')
    print(f'Positive numbers {np.sum(isaac_labels_promising)}')

    current_chunk += 1

    all_info = [quaternions,
                translations,
                promising_indices, 
                splitted_indices, 
                isaac_labels,
                obj_pose_relative,
                current_chunk]
    with open(f'{data_path}/{cat}/{obj_name}_isaac/all_info_{trial}.pkl', 'wb') as f:
        pickle.dump(all_info, f)

    if len(splitted_indices) == current_chunk:
        print('No more')

        np.savez_compressed(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}_rest.npz',
                    quaternions=quaternions,
                    translations=translations,
                    isaac_labels = isaac_labels)

        exit(1)

    exit()



