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
device = torch.device('cuda:1')

custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
    ]

args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)
args.sim_device = device
print(f"using {args.sim_device}")       

def simulate_isaac(quaternions, translations, obj_pose_relative,
                   path_to_obj_mesh, scale, headless, settings):

    isaac = IsaacGymSim(args,headless, settings=settings)
    isaac.init_variables(obj_name=cat,
                         path_to_obj_mesh=path_to_obj_mesh,
                         scale=scale,
                         quaternions=quaternions,
                         translations=translations,
                         obj_pose_relative=obj_pose_relative)
    isaac.create_gripper_asset()
    isaac.create_obj_asset()
    isaac.create_envs(num_envs=isaac.num_envs)

    for i in range(isaac.num_envs):
        env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
        isaac.envs.append(env)

        # ===== Object pose  =====
        isaac.create_obj_actor(env, i, 0)
        idx = i%len(isaac.quaternions)
        isaac.gripper_pose = isaac.get_gripper_pose(isaac.translations[idx], isaac.quaternions[idx])
        isaac.create_gripper_actor(env, isaac.gripper_pose, "panda_gripper", i, 2)
        
    isaac.set_camera_pos()
    isaac.prepare_tensors()
    isaac.execute_grasp()
    isaac.check_grasp_success_pos()
    isaac_labels = isaac.isaac_labels.cpu().numpy()
    isaac.cleanup()
    
    del isaac, env
    gc.collect()
    torch.cuda.empty_cache()

    return isaac_labels


if __name__ == "__main__":
    # initialize global variables
    data_path = 'grasp_data_generated'
    save_path = 'grasp_data_generated'
    path_to_assets = 'assets/grasp_data/' # object mesh and json path

    MAX_ENVS =10000
    headless = True
    trial = 2
    categories = [ 'bottle','box',  'bowl', 'cylinder']

    for cat in categories:
        
        for idx in range(20):

            obj_name = f'{cat}{idx:03}'
            
            if not Path(f'{data_path}/{cat}/{cat}{idx:03}/main1.npz').is_file():
                continue

            if not os.path.exists(f"{data_path}/{cat}/{obj_name}_isaac/"):
                os.mkdir(f"{data_path}/{cat}/{obj_name}_isaac/")

            settings = 0
            if (cat == 'mug') or (cat == "bottle") or (cat == "bowl"):
                settings = 1

            path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
            if settings == 1:
                path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
            scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
            
            if Path(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}_rest.npz').is_file():
                continue

            # get initial run
            fname = f'{data_path}/{cat}/{obj_name}/main{trial}.npz'
            quaternions, translations, obj_pose_relative, is_promising = utils.load_grasp_data(fname)
            indices = np.argwhere(is_promising == 1)
            indices = indices.squeeze()
            if len(indices) == 0:
                continue
            print(f"getting labels for {obj_name}")

            isaac_labels = np.zeros(len(translations))
            isaac_labels_promising = simulate_isaac(quaternions[indices], translations[indices], obj_pose_relative, scale=scale,
                        path_to_obj_mesh=path_to_obj_mesh, headless=headless, settings=settings)

            isaac_labels[indices] = isaac_labels_promising

            
            np.savez_compressed(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}.npz',
                                quaternions=quaternions,
                                translations=translations,
                                isaac_labels = isaac_labels)
            del quaternions, translations

            # run for promising candidates
            indices = indices[isaac_labels_promising==1]
            if len(indices) == 0:
                continue
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
            print(f'Simulating with chunk size {chunk_split_size}')
            splitted_indices = np.array_split(promising_indices, chunk_split_size)

            isaac_labels = np.zeros(len(translations))
            for part_indices in splitted_indices:
                isaac_labels_promising = simulate_isaac(quaternions[part_indices], translations[part_indices], obj_pose_relative, scale=scale,
                            path_to_obj_mesh=path_to_obj_mesh, headless=headless, settings=settings)
                isaac_labels[part_indices] = isaac_labels_promising


                print(f'Simulated: {len(part_indices)}')
                print(f'Positive numbers {np.sum(isaac_labels_promising)}')

            np.savez_compressed(f'{data_path}/{cat}/{obj_name}_isaac/main{trial}_rest.npz',
                            quaternions=quaternions,
                            translations=translations,
                            isaac_labels = isaac_labels)
            



