from curses.ascii import isspace
from curses.panel import bottom_panel
import numpy as np
import json
from pathlib import Path
import time
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
import gc    
from torch.autograd.grad_mode import F
from pathlib import Path
import torch
import utils
from IsaacGymSimulator2_velocity import IsaacGymSim

def load_labelled_data(filename):
    data = np.load(filename)
    qs = []
    ts = []

    for label,q,t in zip(data["isaac_labels"],data['quaternions'],data['translations']):
        if label:
            qs.append(q)
            ts.append(t)
    return qs, ts
    # return data['quaternions'], data['translations']

def load_grasp_data(filename):
    data = np.load(filename)
    is_promising = data['is_promising'].squeeze()
    return data['quaternions'], data['translations'], data['obj_pose_relative'], is_promising

custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
        {'name': '--cat', 'type':str, 'defaut': 'mug'},
        {'name': '--trial', 'type':int, 'defaut': '1'},
        {'name': '--idx', 'type':int, 'defaut': '0'},
        {'name': '--settings', 'type':int, 'defaut': '0'},
        {'name': '--even', 'type':int, 'defaut': 0},
        {'name': '--object', 'type':str, 'defaut': "0"},
        {'name': '--serial_number', 'type':int, 'defaut': 0},
        {'name': '--checkpoint', 'type':int, 'defaut': 0},
        {'name': '--threshold', 'type':int, 'defaut': 0},
    ]

args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)

args.sim_device = torch.device(args.sim_device)
path_to_assets = '../grasper/grasp_data/' # object mesh and json path
print(f"using {args.sim_device}")


cats = ["mug", "bottle","bowl"]
if (args.cat == 'mug') or (args.cat == "bottle") or (args.cat == "bowl"):
    args.settings = 1
else:
    args.settings = 0
# isaac = IsaacGymSim(args,headless=False)
# isaac.num_envs = 40
# counter = 0
# isaac.create_envs(isaac.num_envs)

cat = "bowl"
idx = 19
args.cat = cat #'mug'
grasp_idx = 1
obj_name = f'{cat}{idx:03}'

if (args.cat == 'mug') or (args.cat == "bottle") or (args.cat == "bowl"):
    args.settings = 1
else:
    args.settings = 0

path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.stl'
if args.settings == 1:
    path_to_obj_mesh = f'{path_to_assets}/meshes/{cat}/{obj_name}.obj'
scale = utils.get_scale(f'{path_to_assets}/info/{cat}/{obj_name}.json')

data_path = 'grasp_data_generated'
fname = f'{data_path}/data_labelled/{cat}/{obj_name}_isaac/main1.npz'

quaternions, translations = load_labelled_data(fname)
# translations[grasp_idx][1] += 0.01
# translations[grasp_idx][0] -= 0.02
# translations[grasp_idx][2] += 0.0

fname = f'{data_path}/{cat}/{obj_name}/main1.npz'
_,_,obj_pose_relative,_ = load_grasp_data(fname)
isaac = IsaacGymSim(args,headless=False)
isaac.init_variables(obj_name=args.cat,
                        path_to_obj_mesh=path_to_obj_mesh,
                        scale=scale,
                        quaternions=quaternions[grasp_idx:grasp_idx+1],
                        translations=translations[grasp_idx:grasp_idx+1],
                        obj_pose_relative=obj_pose_relative)
isaac.create_gripper_asset()
isaac.create_obj_asset()
isaac.create_envs(num_envs=1)


for i in range(1):
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