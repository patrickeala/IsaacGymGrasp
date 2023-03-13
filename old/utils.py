import numpy as np
import json
from pathlib import Path
import time
from IsaacGymSimulator2 import IsaacGymSim
from isaacgym import gymutil
from isaacgym.torch_utils import *
import gc    
import torch
from pathlib import Path

custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
        {'name': '--cat', 'type':str, 'defaut': 'mug'},
        {'name': '--trial', 'type':int, 'defaut': '1'},
        {'name': '--idx', 'type':int, 'defaut': '0'},
        {'name': '--settings', 'type':int, 'defaut': '0'},
    ]

args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)
args.sim_device = torch.device(args.sim_device)

print(f"using {args.sim_device}")
if (args.cat == 'mug') or (args.cat == "bottle") or (args.cat == "bowl"):
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

def get_scale(path_to_json):
    with open(path_to_json) as json_file:
        info = json.load(json_file)
        return  info["scale"]

def load_grasp_data(filename):
    data = np.load(filename)
    is_promising = data['is_promising'].squeeze()
    return data['quaternions'], data['translations'], data['obj_pose_relative'], is_promising

def load_labelled_data(filename):
    data = np.load(filename)
    return data['quaternions'], data['translations'], data['isaac_labels']

def load_as_urdf(obj_name=None, obj_path=None, texture_obj_path=None, mass=3, scale=None):

        if not texture_obj_path:
            texture_obj_path = obj_path

        print(f'Scaling object mesh with {scale}.')

        urdf_txt = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
  <link name="{obj_name}">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{texture_obj_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_path}" scale="{scale} {scale} {scale}"/>
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