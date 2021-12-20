from os import device_encoding
from pickle import NONE, TRUE
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
import os

import utils

class IsaacGymSim:
    def __init__(self, args, headless):
        self.start_time = time.time()
        # set random seed
        set_seed(1,True)
        torch.set_printoptions(precision=10, sci_mode=False)

        # acquire gym interface
        self.settings = args.settings
        self.args = args
        self.gym = gymapi.acquire_gym()
        self.device = args.sim_device
        self.num_envs = 0
        self.headless = headless
        self.compute_device_id = self.device.index
        self.graphics_device_id = 0 #self.device.index
        # print(f" !!!!!!!!isaac gym using device id {self.args.compute_device_id} {type(self.args.graphics_device_id)}")
        # create sim
        self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.args.physics_engine, self.get_sim_params())
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if self.headless == False:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")

        self.set_obj_asset_options()
        self.set_gripper_asset_options()
        self.set_object_pose()
    
        # disable gradients
        torch.set_grad_enabled(False)
        
    def init_variables(self, obj_name=None, scale=1, path_to_obj_mesh=None,
                       quaternions=None, translations=None, obj_pose_relative=None):

        assert path_to_obj_mesh is not None,  "Please indicate path to object mesh"

        self.obj_name = obj_name
        self.scale = scale
        self.path_to_obj_mesh = path_to_obj_mesh

        self.translations = translations
        self.quaternions = quaternions
        self.obj_pose_relative = obj_pose_relative
        self.isaac_labels = torch.ones(len(self.translations)).to(self.device)

        # set the flag to detect collision before executing the grasps
        self.detect_collision = True

        if self.num_envs == 0:
            self.num_envs = len(self.translations)
        print(f"The number of environments: {self.num_envs}.")


    def set_camera_pos(self):
        # point camera at middle env
        self.cam_pos = gymapi.Vec3(4, 3, 2)
        self.cam_target = gymapi.Vec3(-4, -3, 0)
        self.middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
        if self.headless == False:
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

    def set_object_pose(self,  translation=[0,0,3], quaternion=[0,0,0,1],):
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

        return gripper_pos

    def execute_grasp(self):
        self.step_simulation(10)
        self.move_gripper_away()
        self.step_simulation(20)
        self.move_obj_to_pos()
        self.step_simulation(20)
        self.move_gripper_to_grasp()
        self.step_simulation(10)
        self.close_gripper()
        self.close_gripper_effort()
        self.gripper_shake()
        self.step_simulation(50)



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
        for i in range(self.num_envs):
            self.root_tensor[2*i, 0] = -self.obj_pose_relative[0]
            self.root_tensor[2*i, 1] = -self.obj_pose_relative[1]
            self.root_tensor[2*i, 2] = -self.obj_pose_relative[2]
        self.gym.set_actor_root_state_tensor(self.sim,  gymtorch.unwrap_tensor(self.root_tensor))
        self.step_simulation(50)
 
        

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
        self.obj_handle = self.gym.create_actor(env, self.obj_asset,self.obj_pos, "random_obj", i, filter) # i means the collision group that the actor belongs to. 3 means bitwise filter for elements in the same collisionGroup to mask off collision
        # get global index of box in rigid body state tensor
        obj_idx = self.gym.get_actor_rigid_body_index(env, self.obj_handle, 0, gymapi.DOMAIN_SIM)
        self.obj_idxs.append(obj_idx)

    def move_gripper_up(self, up = 0.1):
        if self.settings == 0:
            self.pos_action[:,1] = -up
        elif self.settings == 1:
            self.pos_action[:,1] = -up/2
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def move_gripper_down(self, down = 0.1):
        if self.settings == 0:
            self.pos_action[:,1] = -down
        elif self.settings == 1:
            self.pos_action[:,1] = -down/2
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def stop_gripper(self):
        self.pos_action[:,0] = 0
        self.pos_action[:,1] = 0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
    
    def z_translate(self):
        if self.settings == 0:
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
        elif self.settings == 1:
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
        _ss = 20
        if self.settings == 1:
            _ss = 20
            angle /= 2
        self.rotate_gripper(angle/2)
        self.step_simulation(_ss)
        self.rotate_gripper(-angle/2)
        self.step_simulation(_ss)
        self.rotate_gripper(angle/2)
        self.step_simulation(_ss)
        self.rotate_gripper(-angle/2)
        self.step_simulation(_ss)
        self.rotate_gripper(angle/2)
        self.step_simulation(_ss)
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

    def move_gripper_away(self, standoff = 0.5):
        self.pos_action[:,1] = -standoff
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.step_simulation(200)

    def move_gripper_to_grasp(self):
        self.pos_action[:,1] = 0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        self.step_simulation(200)


    def create_gripper_asset(self, assets_dir='assets/urdf/panda_hand', franka_asset_file = "panda_hand.urdf"):
        self.gripper_asset = self.gym.load_asset(self.sim, assets_dir, franka_asset_file, self.gripper_asset_options)
        return self.gripper_asset

    def create_obj_asset(self,obj_name=None):
        asset_dir = 'temp_urdf'
        obj_urdf_file = utils.load_as_urdf(obj_name=self.obj_name, scale=self.scale, obj_path=self.path_to_obj_mesh)
        obj_asset_options = self.obj_asset_options
        self.obj_asset = self.gym.load_asset(self.sim, asset_dir, obj_urdf_file, obj_asset_options)
        return self.obj_asset
        
    def cleanup(self):
        if self.headless == False:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        self.translations = None
        self.quaternions = None

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
            if self.headless == False:

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
        print(f'ARGS Settings: {self.args.settings}')
        if self.settings == 0: # box, cylinder
            obj_asset_options = gymapi.AssetOptions()
            obj_asset_options.disable_gravity = True
            obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            obj_asset_options.vhacd_enabled = False
            # obj_asset_options.convex_decomposition_from_submeshes = True
            # obj_asset_options.vhacd_params = gymapi.VhacdParams()
            # obj_asset_options.vhacd_params.resolution = 1000000
            # obj_asset_options.convex_decomposition_from_submeshes = True
            obj_asset_options.override_com = True
            obj_asset_options.override_inertia = True
            # obj_asset_options.use_mesh_materials = True
            self.obj_asset_options = obj_asset_options
            return obj_asset_options
        elif self.settings == 1:
            obj_asset_options = gymapi.AssetOptions()
            obj_asset_options.disable_gravity = True
            obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            obj_asset_options.vhacd_enabled = True
            # obj_asset_options.convex_decomposition_from_submeshes = True
            obj_asset_options.vhacd_params = gymapi.VhacdParams()
            obj_asset_options.vhacd_params.resolution = 1000000
            # obj_asset_options.convex_decomposition_from_submeshes = True
            obj_asset_options.override_com = True
            obj_asset_options.override_inertia = True
            # obj_asset_options.use_mesh_materials = True
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
            # self.sim_params.physx.max_gpu_contact_pairs = 1048576 * 2 # 1110999
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
