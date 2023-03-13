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
    path_to_assets = 'assets/grasp_data/' # object mesh and json path

    MAX_ENVS = 10000
    headless = True
    trial = utils.args.trial
    cat = utils.args.cat #'mug'
    idx = utils.args.idx
        
    obj_name = f'{cat}{idx:03}'
    

    Path(f"{data_path}/{cat}/{obj_name}_isaac/").mkdir(parents=True, exist_ok=True)
    

    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
    if utils.args.settings == 1:
        path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
    scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    

    # get initial run
    fname = f'{data_path}/{cat}/{obj_name}/main{trial}.npz'
    quaternions, translations, obj_pose_relative, is_promising = utils.load_grasp_data(fname)
    indices = np.argwhere(is_promising == 1)
    indices = indices.squeeze()
    if len(indices) == 0:
        exit(1)
    print(f"getting labels for {obj_name}")

    isaac_labels = np.zeros(len(translations))
    isaac_labels_promising = utils.simulate_isaac(quaternions[indices], translations[indices], obj_pose_relative, scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless)

    isaac_labels[indices] = isaac_labels_promising


    # run for promising candidates
    current_chunk = 0
    indices = indices[isaac_labels_promising==1]
    if len(indices) == 0:
        print('No positive grasps')
        current_chunk = -1

    np.savez_compressed(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}.npz',
                        quaternions=quaternions,
                        translations=translations,
                        isaac_labels = isaac_labels)
    del quaternions, translations

    # gather the data
    quaternions, translations, is_promising = [], [], []
    for _ind in indices:
        _quaternions, _translations, obj_pose_relative, _is_promising = utils.load_grasp_data(fname)
        quaternions.append(_quaternions)
        translations.append(_translations)
        is_promising.append(_is_promising)

    quaternions = np.concatenate(quaternions)
    translations = np.concatenate(translations)
    is_promising = np.concatenate(is_promising)

    promising_indices = np.argwhere(is_promising==1).squeeze()

    chunk_split_size = int( len(promising_indices)/MAX_ENVS ) + 1
    print(f'Total lefover: {len(promising_indices)}')
    #print(f'Simulating with chunk size {chunk_split_size}')
    splitted_indices = np.array_split(promising_indices, chunk_split_size)
    
    isaac_labels = np.zeros(len(translations))
    all_info = [quaternions,
                translations,
                promising_indices, 
                splitted_indices, 
                isaac_labels,
                obj_pose_relative,
                current_chunk]
    with open(f'{data_path}/{cat}/{obj_name}_isaac/all_info.pkl', 'wb') as f:
        pickle.dump(all_info, f)

    exit()
    
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


