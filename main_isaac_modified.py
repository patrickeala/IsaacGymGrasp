import IsaacGymSimulator_modified as isaac_sim
import pickle5 as pickle
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import numpy as np 
import torch as torch
import os
import gc


if __name__ == "__main__":

    # file = "/home/user/isaacgym/python/IsaacGymGrasp/box_mismatch_indcs/box001.npy"
    # with open(file, "rb") as fh:
    #     index_of_grasps = np.load(fh)
    # Add custom arguments
    custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of environments to create"},
        {"name": "--object", "type": str, "default": "", "help": "Object name as in YCB dataset"},
        {"name": "--quality_type", "type": str, "default": "None", "help": "Choose from ['top1', 'top2', 'bottom1', 'bottom2']"},
        {"name": "--headless", "type": str, "default": "Off", "help": "Headless=On has no graphics but faster simulations"},
        {"name": "--dataset", "type": str, "default": "boxes", "help": "'shapenet', 'boxes', 'ycb'"},
    ]
    args = gymutil.parse_arguments(
        description="test",
        custom_parameters=custom_parameters,
    )
    # data_dir = "box_grasps/grasps_tas"
    data_dir = "isaac_test_labels/box/"
    # if args.sim_device == "cuda:0":
    #     data_dir = "box_grasps/grasps_cuda_0"
    #     print("using cuda 0")
    # else:
    #     data_dir = "box_grasps/grasps_cuda_1"
    #     print("using cuda 1")
    category= "box"

    for sample in os.listdir(f'{data_dir}'):
        grasp_file = f'{data_dir}/{sample}'
        filename = grasp_file.split('/')[-1]
        object_name = filename[:6]
        print("grasp_file: ", grasp_file)
        print("filename: ", filename)
        print("object name: ", object_name)        
        isaac = isaac_sim.IsaacGymSim(args)
        print(f"getting labels for {filename}")
        isaac.set_paths(cat=category,obj=object_name,grasps_file=grasp_file)
        isaac.process_grasps(debug_mode=True)

        # ===== Creating gripper asset =====
        isaac.create_gripper_asset()

        # ===== Creating obj asset =====

        isaac.create_obj_asset()

        # isaac.offset = None

        isaac.create_envs(num_envs=isaac.num_envs)
        
        for i in range(isaac.num_envs):
            env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
            isaac.envs.append(env)

            # ===== Object pose  =====
            isaac.create_obj_actor(env, i, 0)
            idx = i%len(isaac.quaternions)
            isaac.gripper_pose = isaac.get_gripper_pose(isaac.translations[idx], isaac.quaternions[idx], isaac.transforms[idx, :, :])
            isaac.create_gripper_actor(env, isaac.gripper_pose, "panda_gripper", i, 2)

        isaac.set_camera_pos()
        isaac.prepare_tensors()
        isaac.execute_grasp()
        isaac.check_grasp_success_pos()
        isaac.save_isaac_labels(filename) 
        # os.remove(grasp_file)       
        isaac.print_results()
        isaac.step_simulation(2)
        isaac.cleanup()

        del isaac
        gc.collect()
        torch.cuda.empty_cache()
        