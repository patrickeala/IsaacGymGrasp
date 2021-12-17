import IsaacGymSimulator_box as isaac_sim
import pickle5 as pickle
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import numpy as np 
import torch as torch
import os
import gc    
        
def get_label_for_file(category,object,file_name,device="cuda:0",headless=False):

    raw_data_path = f"./{object}"
    main_data_file = f"{raw_data_path}/{file_name}"
    labelled_data= f'./{object}_isaac/{file_name}'
    labelled_data_path = f"./{object}_isaac"
    if os.path.exists(labelled_data):
        print("----------------------------------------------------------------------")
        print(f"The Isaac labels for {main_data_file} has already been generated!!!")
        print("Will skip this file")
        print("----------------------------------------------------------------------")
        return 
    
    custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of environments to create"},
        {"name": "--object", "type": str, "default": "", "help": "Object name as in YCB dataset"},
        {"name": "--quality_type", "type": str, "default": "None", "help": "Choose from ['top1', 'top2', 'bottom1', 'bottom2']"},
        {"name": "--dataset", "type": str, "default": "boxes", "help": "'shapenet', 'boxes', 'ycb'"},
    ]

    args = gymutil.parse_arguments(
        description="test",
        custom_parameters=custom_parameters,
    )

    args.sim_device = device


    print(f"using {args.sim_device}")       
    print(f"getting labels for {object}")

    isaac = isaac_sim.IsaacGymSim(args,headless)
    isaac.set_paths(cat=category,obj=object,grasps_file=main_data_file,results_dir=labelled_data_path)
    isaac.process_grasps()

    isaac.create_gripper_asset()
    isaac.create_obj_asset()
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
    isaac.save_isaac_labels() 
    # os.remove(grasp_file)       
    isaac.print_results()
    isaac.cleanup()

    del isaac, env
    gc.collect()
    torch.cuda.empty_cache()

def get_local_grasps_filenames(labelled_data_path,num_of_trial):
    local_grasps_filenames = []

    import numpy as np
    main_data = np.load(f"{labelled_data_path}/main1.npz")
    
    indecies_of_local_grasps = np.where(np.array(main_data["isaac_labels"])==1)[0]

    for index in indecies_of_local_grasps:
        grasps_filename = f"main{num_of_trial}_{index:08}.npz"
        local_grasps_filenames.append(grasps_filename)
    
    return local_grasps_filenames


def get_label_for_object(category,object,num_of_trial,device="cuda:0",headless="off"):
    category= category
    object = object
    num_of_trial = num_of_trial

    labelled_data_path = f'./{object}_isaac'
    if not os.path.exists(labelled_data_path):
        os.mkdir(labelled_data_path)

    get_label_for_file(category,object,f"main{num_of_trial}.npz",device=device,headless=headless)

    local_grasps_filenames = get_local_grasps_filenames(labelled_data_path,num_of_trial)
    for local_grasps_filename in local_grasps_filenames:
        get_label_for_file(category,object,local_grasps_filename,device=device,headless=headless)
    
    print(f"done with labelling object {object}")

if __name__ == "__main__":
    category = "box"
    for i in range(20):
        get_label_for_object(category=category,object=f"box{i:03}",num_of_trial=1,device="cuda:1",headless=False)
