from curses.ascii import isspace
import numpy as np
import json
from pathlib import Path
import time
from IsaacGymSimulator2_velocity import IsaacGymSim
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
import gc    
from torch.autograd.grad_mode import F
from pathlib import Path
import torch


custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
        {'name': '--cat', 'type':str, 'default': 'mug'},
        {'name': '--trial', 'type':int, 'default': '1'},
        {'name': '--idx', 'type':int, 'default': '0'},
        {'name': '--settings', 'type':int, 'default': '0'},
        {'name': '--even', 'type':int, 'default': 0},
        {'name': '--object', 'type':str, 'default': "0"},
        {'name': '--experiment_number', 'type':int, 'default': 0},
        {'name': '--checkpoint', 'type':int, 'default': 0},
        {'name': '--threshold', 'type':int, 'default': 0},
        {'name': '--exp_dir', 'type':str, 'default': 'experiment_table'},
        {'name': '--mode', 'type':str, 'default': 'refinement'},

    ]

args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)

args.sim_device = torch.device(args.sim_device)

print(f"using {args.sim_device}")
if (args.cat == 'mug') or (args.cat == "bottle") or (args.cat == "bowl")or (args.cat == "fork")or (args.cat == "hammer")or (args.cat == "scissor")or (args.cat == "pan")or (args.cat == "spatula"):
  args.settings = 1
else:
  args.settings = 0


def simulate_isaac(quaternions, translations, obj_pose_relative,
                   path_to_obj_mesh, scale, headless):

    isaac = IsaacGymSim(args,headless)
    isaac.init_variables(obj_name=args.cat,
                         path_to_obj_mesh=path_to_obj_mesh,
                         scale=scale,
                         quaternions=quaternions,
                         translations=translations,
                         obj_pose_relative=obj_pose_relative,
                         )
    isaac.create_gripper_asset()
    isaac.create_obj_asset()
    isaac.create_envs(num_envs=isaac.num_envs)

    
    for i in range(isaac.num_envs):
        env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
        isaac.envs.append(env)

        # 0 0, 0 1, 1 1, 1 0,
        # ===== Object pose  =====
        isaac.create_obj_actor(env, i, 0)
        #isaac.create_gripper_actor(env, isaac.gripper_pose, "panda_gripper", i, 0)

        idx = i%len(isaac.quaternions)
        isaac.gripper_pose = isaac.get_gripper_pose(isaac.translations[idx], isaac.quaternions[idx])
        isaac.create_gripper_actor(env, isaac.gripper_pose, "panda_gripper", i, 0)

        # isaac.create_gripper_actor(env, isaac.gripper_pose, "panda_gripper", i, 2)
        #num_bodies = isaac.gym.get_actor_rigid_body_count(env, isaac.gripper_handle)
        #print(f'gripper_num_bodies {num_bodies}')
        #num_bodies = isaac.gym.get_actor_rigid_body_count(env, isaac.obj_handle)
        #print(f'object_num_bodies {num_bodies}')
        
    isaac.set_camera_pos()
    isaac.prepare_tensors()
    isaac.execute_grasp()
    # isaac.step_simulation(1000)
    isaac.check_grasp_success_pos()
    # isaac.step_simulation(1000)
    isaac_labels = isaac.isaac_labels.cpu().numpy()
    


    isaac.cleanup()
    
    del isaac, env
    gc.collect()
    torch.cuda.empty_cache()

    return isaac_labels

def get_scale(path_to_json):
    with open(path_to_json) as json_file:
        info = json.load(json_file)
        return  info["scale"]

def load_grasp_data(filename):
    data = np.load(filename)
    is_promising = data['is_promising'].squeeze()
    return data['quaternions'], data['translations'], data['obj_pose_relative'], is_promising

def load_experiment_data(filename):
    data = np.load(filename)
    return data['quaternions'], data['translations'], data['obj_pose_relative']

def load_labelled_data(filename):
    data = np.load(filename)
    return data['quaternions'], data['translations'], data['isaac_labels']

def load_as_urdf(obj_name=None, obj_path=None, texture_obj_path=None, mass=0.2, scale=None):   # box 0.8  mug 0.4

        mass = 0.2


        if not texture_obj_path:
          texture_obj_path = obj_path

        # unify scale parameter to [x,y,z] scaling
        if not isinstance(scale,list):
          scale = [scale for _ in range(3)]
        print(f'Scaling object mesh with {scale}.')
        print(f"mass of {obj_name} is {mass}")
        urdf_txt = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="{obj_name}">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{texture_obj_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="0.0000" ixy="0.0" ixz="0.0" iyy="0.000" iyz="0.0" izz="0.0000"/>
    </inertial>
  </link>
</robot>
    """

        save_file = f'{obj_name}.urdf'
        Path('temp_urdf').mkdir(parents=True, exist_ok=True)
        save_path = f'temp_urdf/{save_file}'
        urdf_file = open(save_path, 'w')
        urdf_file.write(urdf_txt)

        return save_file