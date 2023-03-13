import pickle
import numpy as np
from numpy.core.numeric import argwhere
from requests import head
# import utils_fakeworld as utils
import utils_fakeworld as utils
import time
from IsaacGymSimulator2_fakeworld import IsaacGymSim
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
    data_path = '../grasper/grasp_data_generated'
    save_path = 'fakeworld_grasp_data_table'
    # save_path = 'fakeworld_grasp_data_generated_friction'
    # save_path = 'fakeworld_grasp_data_generated_longfinger'
    path_to_assets = '../grasper/grasp_data/' # object mesh and json path



    MAX_ENVS = 100
    # headless = True
    headless = False
    trial = utils.args.trial
    cat = utils.args.cat #'mug'
    idx = utils.args.idx

    # headless = True
    # headless = False
    # trial = 1
    # idx = 0


    obj_name = f'{cat}{idx:03}'
    

    Path(f"{data_path}/{cat}/{obj_name}_isaac/").mkdir(parents=True, exist_ok=True)
    Path(f"{save_path}/{cat}/{obj_name}_isaac_sim2fake/").mkdir(parents=True, exist_ok=True)
    

    # Get object mesh
    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
    if utils.args.settings == 1:
        path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
    scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')
    

    # get initial run
    if utils.args.even == 0:
        fname = f'{data_path}/{cat}/{obj_name}/main{trial}.npz'
        # main_save_name = f'{save_path}/{cat}/{obj_name}_isaac/main{trial}.npz'
    # elif utils.args.even == 1:
        # fname = f'{data_path}/{cat}/{obj_name}/main{trial}_even_grasps.npz'
        # main_save_name = f'{save_path}/{cat}/{obj_name}_isaac/main{trial}_even_grasps.npz'



    quaternions, translations, obj_pose_relative, is_promising = utils.load_grasp_data(fname)
    # print(is_promising)
    indices = np.argwhere(is_promising == 1)
    indices = indices.squeeze()[:MAX_ENVS]
    # indices = indices.squeeze()[:5]
    if len(indices) == 0:
        print("no promising grasps")
        exit(1)
    print(f"getting labels for {obj_name}")


    # friction_sim = 0.25
    # friction_fake = 0.06


    # USE THIS FOR STRICT FAKE WORLD ========================
    friction_sim = 0.08
    # friction_fake = 0.06

    if cat == "bowl":
        friction_sim = 0.2
    # if cat == 'scissor':
    #     friction_sim = 0.12
    # if cat == 'fork':
    #     friction_sim = 0.08
    # if cat == 'hammer':
    #     friction_sim = 0.08
    # if cat == 'pan':
    #     friction_sim = 0.15
    # ==============================================


    sim_close_vel = -0.1
    # fake_close_vel = -0.07

    # print(indices)
    # exit()

    # indices = [121]
    # print(indices)
    # exit()

    # print(quaternions[indices])
    # print(translations[indices])
    # exit()
    
    # for box0
    # quat = np.array([[0, 0.707, 0, -0.707]])
    # trans = np.array([[0.15, -0.04825146, -0.00177786]])

    # for mug0
    # quat = quaternions[indices]
    # trans = translations[indices]

    print(len(quaternions))
    print(len(translations))
    # exit()
    # indices = np.arange(1)
    # headless=False


    isaac_fake_labels = np.zeros(len(translations))
    isaac_sim_labels = np.zeros(len(translations))

    # fake_labels = utils.simulate_isaac(quaternions[indices], translations[indices], obj_pose_relative, long_gripper=False, gravity=False, table=True, scale=scale,
    #             path_to_obj_mesh=path_to_obj_mesh, headless=headless, friction=friction_sim, close_vel=sim_close_vel)
    # isaac_fake_labels[indices] = fake_labels



    sim_labels = utils.simulate_isaac(quaternions[indices], translations[indices], obj_pose_relative, long_gripper=False, gravity=False, table=False, scale=scale,
                path_to_obj_mesh=path_to_obj_mesh, headless=headless, friction=friction_sim, close_vel=sim_close_vel)
    isaac_sim_labels[indices] = sim_labels




    

    # idxs_to_check = np.logical_and(long_finger_labels==0, short_finger_labels==1)
    # print(indices[idxs_to_check])
    # exit()

    # print("\n\n\n===============")
    # print(f"original n_grasps: {len(translations)}")
    # print(f"promising: {len(indices)}")
    # print(f"sim+: {np.sum(isaac_sim_labels)}")
    # print(f"fake+: {np.sum(isaac_fake_labels)}")
    # # print(f"len of fake labels : {isaac_fake_labels}")
    # print("===============\n\n\n")



    # all_info = [quaternions,
    #             translations,
    #             isaac_sim_labels,
    #             isaac_fake_labels,
    #             obj_pose_relative]

    # if utils.args.save == 1:
    #     with open(f'{save_path}/{cat}/{obj_name}_isaac_sim2fake/all_info_{trial}.pkl', 'wb') as f:
    #         pickle.dump(all_info, f)
    # else:
    #     print("not saving")

    # with open(f'{save_path}/{cat}/{obj_name}_isaac_sim2fake/all_info_{trial}.pkl', 'wb') as f:
        # pickle.dump(all_info, f)

    # print("==========================\n\n\n")
    exit()




