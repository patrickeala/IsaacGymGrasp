from os import device_encoding
from pickle import NONE
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time

from scipy.spatial.transform import Rotation as R
import numpy as np
import os


class IsaacGymSim:
    def __init__(self, args):
        self.start_time = time.time()
        # set random seed
        set_seed(1,True)
        torch.set_printoptions(precision=10, sci_mode=False)

        # acquire gym interface
        self.args = args
        self.gym = gymapi.acquire_gym()
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.num_envs = self.args.num_envs
        self.quality_type = self.args.quality_type


        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.get_sim_params())
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if self.args.headless == 'Off':
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

        self.set_obj_asset_options()
        self.set_gripper_asset_options()
        self.set_object_pose()

    def set_paths(self,cat,obj,grasps_file,results_dir='isaac_test_labels',obj_file=None,is_shapenet=False):
        # cat -- category of the object, eg "box"
        # obj -- name of the object, eg "box001"
        # obj_file -- the name of the object.obj
        # grasps_file -- the path of the file contains grasps to load
        # results_dir -- the directory to store the labelled grasps file
        self.obj = obj
        self.cat = cat
        if not obj_file:
            obj_file = f"{obj}.stl" if not is_shapenet else f"{obj}.obj"
        self.obj_file = obj_file
        self.obj_path = f"assets/{self.cat}/{self.obj_file}"
        self.grasp_file = grasps_file
        self.results_dir = results_dir

    def set_camera_pos(self):
        # point camera at middle env
        self.cam_pos = gymapi.Vec3(4, 3, 2)
        self.cam_target = gymapi.Vec3(-4, -3, 0)
        self.middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        if self.args.headless == 'Off':
            self.gym.viewer_camera_look_at(self.viewer, self.middle_env, self.cam_pos, self.cam_target)

    def create_envs(self, num_envs=256):
        self.num_per_row = int(math.sqrt(num_envs))
        self.spacing = 1.0
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)


        print("Creating %d environments" % num_envs)


        # ===== add ground plane  =====
        self.add_ground_plane()



    def prepare_tensors(self):
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim) #Prepares simulation with buffer allocations

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

        #Set action tensors
        self.pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        self.pos_action[:,-2:] = 0.04
        self.effort_action = torch.zeros_like(self.pos_action)

        # get contact force tensor 
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_contact_forces =  gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1)

    def add_ground_plane(self, z=-1):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0,0,1)
        plane_params.distance = -z
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        self.obj_idxs = []
        self.gripper_idxs = []


    def set_object_pose(self,  translation=[0,0,1], quaternion=[0,0,0,1],):
        obj_pose = gymapi.Transform()
        obj_pose.p.x = translation[0]
        obj_pose.p.y = translation[1]
        obj_pose.p.z = translation[2]

        obj_pose.r = gymapi.Quat()
        obj_pose.r.x = quaternion[0]
        obj_pose.r.y = quaternion[1]
        obj_pose.r.z = quaternion[2]
        obj_pose.r.w = quaternion[3]
        self.obj_pos = obj_pose
        return obj_pose


    def get_gripper_pose(self, translation=[0,0,0], quaternion=[0,0,0,1], transform=None):
        gripper_pose = gymapi.Transform()
        gripper_pose.p = gymapi.Vec3(0, 0, 0)
        gripper_pose.p.x = translation[0] 
        gripper_pose.p.y = translation[1]
        gripper_pose.p.z = translation[2]

        gripper_pose.r = gymapi.Quat()
        gripper_pose.r.x = quaternion[0]
        gripper_pose.r.y = quaternion[1]
        gripper_pose.r.z = quaternion[2]
        gripper_pose.r.w = quaternion[3]
        # rotate the gripper around z axis for 90 degree
        flipping_rot = gymapi.Quat()
        flipping_rot.w = np.cos(-np.pi/4)
        flipping_rot.z = np.sin(-np.pi/4)

        gripper_pose.r = gripper_pose.r * flipping_rot

        return gripper_pose

    def print_results(self):
        # if self.debug_mode:
        #     print("the imdex of successful grasps: ", np.where(np.array(self.data['qualities_1'])== 0)[0][np.where(torch.Tensor.cpu(self.isaac_labels)==1)[0]])
        print(f"\nObj: {self.cat}\nQuality Type: {self.quality_type}\nSuccessful Grasps: {torch.count_nonzero(self.isaac_labels)}/{self.isaac_labels.size()[0]}\nPercentage: {torch.mean(self.isaac_labels).item()*100}%")
        print("--- %s seconds ---" % (time.time() - self.start_time))

    def execute_grasp(self):
        self.step_simulation(50)
        self.move_gripper_away()
        self.step_simulation(20)
        self.move_obj_to_pos()
        self.step_simulation(20)
        self.move_gripper_to_grasp()
        self.step_simulation(50)
        self.close_gripper()
        self.close_gripper_effort()
        self.gripper_shake()
        self.step_simulation(100)


    def save_isaac_labels(self, filename=None, checker_mode=False):
        import pickle
        import os
        file_dir = f'{self.results_dir}/{self.cat}'
        if not os.path.exists(file_dir):
           os.makedirs(file_dir)
        if checker_mode:
            index_to_modify = self.index_of_grasps[np.where( np.array(self.isaac_labels.cpu())== 0)[0]]
            print("checker mode up and the index to redefine: ", index_to_modify)
            self.data['isaac_labels'][index_to_modify] = 0
        else:
            self.data['isaac_labels'] = np.array(self.isaac_labels.cpu())
        if not filename:
            filename = f"{file_dir}/{self.obj}.pkl"
        else:
            filename = f"{file_dir}/{filename}"
        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def gripper_finger_stable(self, vel_threshold = 0.01):
        f1_temp = self.dof_pos[:,2,0].clone().detach()
        f2_temp = self.dof_pos[:,3,0].clone().detach()
        labels = torch.ones(self.num_envs).to(self.device)
        for _ in range(10):
            unstable = torch.logical_or(torch.ne(f1_temp, self.dof_pos[:,2,0]),
                                        torch.ne(f2_temp, self.dof_pos[:,3,0]))
            labels = torch.where(unstable, torch.zeros(1).to(self.device), labels)
            self.step_simulation(1)
        return labels


    def check_grasp_success_vel(self, vel_threshold = 0.01):
        finger_vel =  (self.dof_vel[:,2,0] + self.dof_vel[:,3,0])/2
        labels = torch.where(finger_vel > vel_threshold, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))
        return labels

    def check_grasp_success_pos(self, gap_threshold = 0.0005):
        # success if the gripper fingers are not close together
        gripper_gap = self.dof_pos[:,2,0] + self.dof_pos[:,3,0]
        labels = torch.where(gripper_gap > gap_threshold, 1.0, 0.0) 
        # print("labels of grasps holding the object in the end: ", labels)
        # print("labels of grasps without collision during approaching: ", self.isaac_labels)
        self.isaac_labels = self.isaac_labels * labels
        return self.isaac_labels

    def get_root_tensor(self):
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor).to(self.device)
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_angvels = self.root_tensor[:, 10:13]

    def move_obj_to_pos(self, x=0,y=0,z=0):
        print("---------------")
        print(f"moving obj to {-self.obj_pose_relative[0],-self.obj_pose_relative[1],-self.obj_pose_relative[2]}")
        for i in range(self.num_envs):
            self.root_tensor[2*i, 0] = -self.obj_pose_relative[0]
            self.root_tensor[2*i, 1] = -self.obj_pose_relative[1]
            self.root_tensor[2*i, 2] = -self.obj_pose_relative[2]
        self.gym.set_actor_root_state_tensor(self.sim,  gymtorch.unwrap_tensor(self.root_tensor))
        self.step_simulation(50)

    def process_grasps(self, file=None, index_of_grasps=np.array([]), debug_mode = False):
        import pickle5 as pickle
        if not file:
            file = self.grasp_file
        with open(file, "rb") as fh:
            self.data = pickle.load(fh)
        print("obj_relative: ", self.data['obj_pose_relative'])
        # file_to_read = open(file, "rb")
        # self.data = pickle.load(file_to_read)
        self.index_of_grasps = index_of_grasps
        flag = not np.any(self.index_of_grasps)
        # if the index_of_grasps is empty, namely no grasps to filter out 
        self.debug_mode = debug_mode
        if flag and not self.debug_mode:
            self.obj_name = self.data['fname']
            # self.full_pc = self.data['full_pc']
            self.transforms = self.data['transforms']
            self.translations = self.data['translations']
            self.quaternions = self.data['quaternions']
            self.collisions = self.data['collisions']
            self.qualities_1 = np.array(self.data['qualities_1'])
            self.qualities_2 = np.array(self.data['qualities_2'])
            self.obj_pose_relative = self.data['obj_pose_relative']
            try:
                self.scale = self.data['scale']
            except:
                self.scale = 1
 
        else:
            if self.debug_mode:
                # print(np.where(self.qualities_1 != 0)[0])
                # self.index_of_grasps = np.where(np.array(self.data['qualities_1'])== 0)[0]
                self.index_of_grasps = np.where(np.array(self.data['isaac_labels'])== 1)[0]

                # index_of_grasps = [2544]
            else:
                self.index_of_grasps = self.index_of_grasps.flatten()
            self.obj_name = self.data['fname']
            # self.full_pc = self.data['full_pc']
            self.transforms = self.data['transforms'][self.index_of_grasps]
            self.translations = self.data['translations'][self.index_of_grasps]
            self.quaternions = self.data['quaternions'][self.index_of_grasps]
            self.collisions = self.data['collisions'][self.index_of_grasps]
            self.qualities_1 = np.array(self.data['qualities_1'])[self.index_of_grasps]
            self.qualities_2 = np.array(self.data['qualities_2'])[self.index_of_grasps]
            self.obj_pose_relative = np.array(self.data['obj_pose_relative'])
            try:
                self.scale = self.data['scale']
            except:
                self.scale = 1
        print("---------------------------")
        print("scaling the object with :", self.scale)
        print("---------------------------")
        # self.isaac_labels = np.ones_like(self.qualities_1)
        self.isaac_labels = torch.ones(len(self.qualities_2)).to(self.device)
        # set the flag to detect collision before executing the grasps
        self.detect_collision = True
        # print("labels: ", self.isaac_labels)

        # print(np.where(self.qualities_1 != 0)[0])
        # print(self.qualities_1[900])
        # print(self.qualities_2[900])
        # print(index_of_grasps[900])
        # exit()
        if self.num_envs == 0:
            self.num_envs = len(self.qualities_1)
        print("the number of environments: ", self.num_envs)
        print("the qualities_1 are all zeros: ",(not np.any(self.qualities_1)))
        print("the qualities_2 are all zeros: ",(not np.any(self.qualities_2)))

    def detect_collision_during_approaching(self):
        if self.detect_collision:
            index_of_env_in_collision = torch.where((self.net_contact_forces.any(dim = 1)))
            self.isaac_labels[index_of_env_in_collision] = 0
        return

    def get_gripper_pose(self, translation=[0,0,0], quaternion=[0,0,0,1], transform=None):

        gripper_pose = gymapi.Transform()
        gripper_pose.p = gymapi.Vec3(0, 0, 0)
        gripper_pose.p.x = translation[0] 
        gripper_pose.p.y = translation[1]
        gripper_pose.p.z = translation[2]

        gripper_pose.r = gymapi.Quat()
        gripper_pose.r.x = quaternion[0]
        gripper_pose.r.y = quaternion[1]
        gripper_pose.r.z = quaternion[2]
        gripper_pose.r.w = quaternion[3]

        flipping_rot = gymapi.Quat()
        flipping_rot.w = np.cos(-np.pi/4)
        flipping_rot.z = np.sin(-np.pi/4)
        gripper_pose.r = gripper_pose.r * flipping_rot
        return gripper_pose

    def create_gripper_actor(self, env, pose, name, i, filter = 0):

        franka_dof_props = self.gym.get_asset_dof_properties(self.gripper_asset)
        franka_upper_limits = franka_dof_props["upper"]
        franka_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:].fill(800.0)
        franka_dof_props["damping"][:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(self.gripper_asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32) # initializing the joints degrees
        # grippers open
        default_dof_pos[-2:] = franka_upper_limits[-2:]
        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos
        self.gripper_handle = self.gym.create_actor(env,self.gripper_asset, pose, name, i, filter)

        # set dof properties
        self.gym.set_actor_dof_properties(env, self.gripper_handle, franka_dof_props)

        # set initial dof states
        self.gym.set_actor_dof_states(env, self.gripper_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        self.gym.set_actor_dof_position_targets(env, self.gripper_handle, default_dof_pos)

        gripper_idx = self.gym.find_actor_rigid_body_index(env, self.gripper_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.gripper_idxs.append(gripper_idx)

    def create_obj_actor(self, env, i, filter = 0):
        self.obj_handle = self.gym.create_actor(env, self.obj_asset,self.obj_pos, self.obj , i, filter) # i means the collision group that the actor belongs to. 3 means bitwise filter for elements in the same collisionGroup to mask off collision
        # get global index of box in rigid body state tensor
        obj_idx = self.gym.get_actor_rigid_body_index(env, self.obj_handle, 0, gymapi.DOMAIN_SIM)
        self.obj_idxs.append(obj_idx)

    def move_gripper_up(self, up = 0.1):
        self.pos_action[:,1] = -up
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def move_gripper_down(self, down = 0.1):
        self.pos_action[:,1] = down
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def stop_gripper(self):
        self.pos_action[:,0] = 0
        self.pos_action[:,1] = 0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
    
    def z_translate(self):
        self.move_gripper_up()
        self.step_simulation(15)
        self.move_gripper_down()
        self.step_simulation(15)
        self.move_gripper_up()
        self.step_simulation(15)
        self.move_gripper_down()
        self.step_simulation(15)
        self.move_gripper_up()
        self.step_simulation(15)
        self.move_gripper_down()
        self.step_simulation(15)
        self.stop_gripper()
        self.step_simulation(15)

    def rotate_gripper(self, angle = np.pi/2):
        self.pos_action[:,0] = angle
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def y_rotate(self, angle =2* np.pi):
        self.rotate_gripper(angle/2)
        self.step_simulation(100)
        self.rotate_gripper(-angle/2)
        self.step_simulation(100)
        self.rotate_gripper(angle/2)
        self.step_simulation(100)
        self.rotate_gripper(-angle/2)
        self.step_simulation(100)
        self.rotate_gripper(angle/2)
        self.step_simulation(100)
        self.stop_gripper()
        self.step_simulation(30)

    def gripper_shake(self):
        self.z_translate()
        self.y_rotate()
        self.step_simulation(50)

    def close_gripper_effort(self, effort = -100):
        for _ in range(200):
            self.effort_action[:,2] = effort
            self.effort_action[:,3] = effort
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))
            self.step_simulation(1)
        self.effort_action[:,2] = 0
        self.effort_action[:,3] = 0
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))

    def close_gripper(self):
        self.detect_collision = False
        self.step_simulation(10)

        self.pos_action[:,2] = 0
        self.pos_action[:,3] = 0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.step_simulation(200)

    def move_gripper_away(self, standoff = 0.25):
        self.pos_action[:,1] = -standoff
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.step_simulation(100)

    def move_gripper_to_grasp(self):
        self.pos_action[:,1] = 0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.step_simulation(200)


    def create_gripper_asset(self, assets_dir='assets/urdf/panda_hand', franka_asset_file = "panda_hand.urdf"):
        self.gripper_asset = self.gym.load_asset(self.sim, assets_dir, franka_asset_file, self.gripper_asset_options)
        return self.gripper_asset

    def create_obj_asset(self,obj_name=None):
        asset_dir = 'temp_urdf'
        obj_urdf_file = self.load_as_urdf(obj_name=None)
        obj_asset_options = self.obj_asset_options
        self.obj_asset = self.gym.load_asset(self.sim, asset_dir, obj_urdf_file, obj_asset_options)
        return self.obj_asset
        
    def cleanup(self):
        if self.args.headless == 'Off':
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step_simulation(self, num_steps=1000):
        for _ in range(num_steps):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.detect_collision_during_approaching()

            # update viewer
            if self.args.headless == 'Off':

                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

            

            self.get_root_tensor()

            self.obj_pos = self.rb_states[self.obj_idxs, :3]
            self.obj_rot = self.rb_states[self.obj_idxs, 3:7]
            self.obj_vel = self.rb_states[self.obj_idxs, 7:]
            self.gripper_pos = self.rb_states[self.gripper_idxs, :3]
            self.gripper_rot = self.rb_states[self.gripper_idxs, 3:7]
            self.gripper_vel = self.rb_states[self.gripper_idxs, 7:]   
            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)

    def set_obj_asset_options(self):
        obj_asset_options = gymapi.AssetOptions()
        obj_asset_options.disable_gravity = True
        # obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
        obj_asset_options.vhacd_enabled = True
        obj_asset_options.vhacd_params = gymapi.VhacdParams()
        obj_asset_options.vhacd_params.resolution = 1000000

        # obj_asset_options.convex_decomposition_from_submeshes = True
        obj_asset_options.override_com = True
        obj_asset_options.override_inertia = True
        obj_asset_options.use_mesh_materials = True
        self.obj_asset_options = obj_asset_options
        return obj_asset_options

    def set_gripper_asset_options(self):
        gripper_asset_options = gymapi.AssetOptions()
        gripper_asset_options.armature = 0.4
        gripper_asset_options.fix_base_link = True
        gripper_asset_options.disable_gravity = True
        gripper_asset_options.flip_visual_attachments = False
        self.gripper_asset_options = gripper_asset_options
        return gripper_asset_options

    def load_as_urdf(self, obj_name=None, asset_dir=None, obj_path=None, texture_obj_path=None, mass=1.5, scale=None):
        if not scale:
            scale = self.scale
        if not obj_path:
            obj_path = self.obj_path
        if not texture_obj_path:
            texture_obj_path = obj_path
        if not obj_name:
            obj_name = self.obj
                
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
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>
</robot>
    """
        save_file = f'{obj_name}.urdf'
        save_path = f'temp_urdf/{save_file}'
        urdf_file = open(save_path, 'w')
        urdf_file.write(urdf_txt)

        return save_file

    def get_sim_params(self):
        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 4
        self.sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        if self.args.physics_engine == gymapi.SIM_PHYSX:
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 8
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.rest_offset = 0.0
            self.sim_params.physx.contact_offset = 0.001
            self.sim_params.physx.friction_offset_threshold = 0.001
            self.sim_params.physx.friction_correlation_distance = 0.0005
            self.sim_params.physx.num_threads = self.args.num_threads
            self.sim_params.physx.use_gpu = self.args.use_gpu
        else:
            raise Exception("This example can only be used with PhysX")
        return self.sim_params

def set_seed(seed, torch_deterministic=False):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed
