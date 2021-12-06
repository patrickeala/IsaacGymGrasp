from os import device_encoding
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
        np.random.seed(42)
        torch.set_printoptions(precision=4, sci_mode=False)
        # torch.use_deterministic_algorithms(True)

        # acquire gym interface
        self.args = args
        self.gym = gymapi.acquire_gym()
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'
        self.num_envs = self.args.num_envs


        # create sim
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, self.args.physics_engine, self.get_sim_params())
        if self.sim is None:
            raise Exception("Failed to create sim")

        # create viewer
        if self.args.headless == 'Off':
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise Exception("Failed to create viewer")



        # ===== next lines executed in main =====


        # ===== process grasps  =====
        # self.set_paths()
        # self.process_grasps(self.grasp_file)







        # self.create_envs(num_envs=self.num_envs)
        # self.set_camera_pos()

        # self.prepare_tensors()
        # self.step_simulation(2000)

        # self.execute_grasp()
        # self.isaac_labels = self.check_grasp_success_pos()
        # self.save_isaac_labels()        
        # self.print_results()
        # self.step_simulation(200)
        # self.cleanup()


    def set_paths(self):
        self.cat = self.args.object
        self.quality_type = self.args.quality_type
        self.asset_root = "../../assets"
        self.dataset = self.args.dataset

        if self.dataset == 'boxes':
            self.obj_path = f"boxes/{self.cat}.stl" 
            self.obj_asset_root = "../../../GRASP/grasper/assets"
            self.test_dir = 'box_grasps/grasps'

        elif self.dataset == 'ycb':
            self.obj_path = f"ycb/meshes_10_objects/{self.cat}/google_16k/textured.obj" 
            self.obj_asset_root = "../../assets"
            self.test_dir = 'isaac_test_10_meshes'

        elif self.dataset == 'shapenet':
            # self.obj_path = f"ycb/meshes_10_objects/{self.cat}/google_16k/textured.obj" 
            self.obj_asset_root = "../../../GRASP/"
            # self.test_dir = 'isaac_test_10_meshes'

            
        if self.dataset == 'shapenet':
            pass
        else:
            if self.quality_type == 'None':
                self.grasp_file = f"{self.test_dir}/{self.cat}.pkl"
            else:
                self.grasp_file = f"{self.test_dir}/{self.cat}_{self.quality_type}.pkl"
                print("====")

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

    def add_ground_plane(self, z=-1):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0,0,1)
        plane_params.distance = -z
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        self.obj_idxs = []
        self.gripper_idxs = []


    def get_object_pose(self,  translation=[0,0,1], quaternion=[0,0,0,1],):
        obj_pose = gymapi.Transform()
        obj_pose.p.x = translation[0]
        obj_pose.p.y = translation[1]
        obj_pose.p.z = translation[2]

        obj_pose.r = gymapi.Quat()
        obj_pose.r.x = quaternion[0]
        obj_pose.r.y = quaternion[1]
        obj_pose.r.z = quaternion[2]
        obj_pose.r.w = quaternion[3]
        return obj_pose


    def get_gripper_pose(self, translation=[0,0,0], quaternion=[0,0,0,1], transform=None):

        # gripper_pose = gymapi.Transform()
        # gripper_pose.p = gymapi.Vec3(0, 0, 0)
        # gripper_pose.p.x = -0.01
        # gripper_pose.p.y = -0.01
        # # gripper_pose.p.z = self.offset + 0.125
        # gripper_pose.p.z = self.offset + 0.18
        # gripper_pose.r = gymapi.Quat.from_euler_zyx(np.pi, 0, 0)

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

    def print_results(self):
        print(f"\nObj: {self.cat}\nQuality Type: {self.quality_type}\nSuccessful Grasps: {torch.count_nonzero(self.isaac_labels)}/{self.isaac_labels.size()[0]}\nPercentage: {torch.mean(self.isaac_labels).item()*100}%")
        print("--- %s seconds ---" % (time.time() - self.start_time))

    def execute_grasp(self):
        self.step_simulation(50)
        self.move_gripper_away()
        self.move_obj_to_pos()
        self.move_gripper_to_grasp()
        self.close_gripper()
        self.gripper_shake()
        self.step_simulation(100)


    def save_isaac_labels(self, results_dir = 'isaac_test_labels'):
        import pickle
        import os
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.data['isaac_labels'] = np.array(self.isaac_labels.cpu())
        filename = f"{results_dir}/{self.cat}.pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def gripper_finger_stable(self, vel_threshold = 0.01):
        f1_temp = self.dof_pos[:,2,0].clone().detach()
        f2_temp = self.dof_pos[:,3,0].clone().detach()
        labels = torch.ones(self.num_envs).to(self.device)
        for _ in range(10):
            # print(f"f1_temp: {f1_temp}")
            # print(f"f2_temp: {f2_temp}")
            # print(f"self.dof_pos[:,2,0]: {self.dof_pos[:,2,0]}")
            # print(f"self.dof_pos[:,3,0]: {self.dof_pos[:,3,0]}")
            unstable = torch.logical_or(torch.ne(f1_temp, self.dof_pos[:,2,0]),
                                        torch.ne(f2_temp, self.dof_pos[:,3,0]))
            # print(f"torch.ne(f1_temp, self.dof_pos[:,2,0]): {torch.ne(f1_temp, self.dof_pos[:,2,0])}")
            # print(f"torch.ne(f2_temp, self.dof_pos[:,3,0]): {torch.ne(f2_temp, self.dof_pos[:,3,0])}")
            # print(f"unstable: {unstable}")
            labels = torch.where(unstable, torch.zeros(1).to(self.device), labels)
            # print(f"labels: {labels}")
            # print("=======================")

            # f1_temp = self.dof_pos[:,2,0].clone().detach()
            # f2_temp = self.dof_pos[:,3,0].clone().detach()


            self.step_simulation(1)
        # print(labels)
        return labels


    def check_grasp_success_vel(self, vel_threshold = 0.01):
        finger_vel =  (self.dof_vel[:,2,0] + self.dof_vel[:,3,0])/2
        labels = torch.where(finger_vel > vel_threshold, torch.ones(1).to(self.device), torch.zeros(1).to(self.device))
        return labels

    def check_grasp_success_pos(self, gap_threshold = 0.0005):
        # success if the gripper fingers are not close together
        gripper_gap = self.dof_pos[:,2,0] + self.dof_pos[:,3,0]
        labels = torch.where(gripper_gap > gap_threshold, 1.0, 0.0)

        return labels

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
            # self.root_tensor[2*i, 0] = 0
            # self.root_tensor[2*i, 1] = 0
            # self.root_tensor[2*i, 2] = 0    
        self.gym.set_actor_root_state_tensor(self.sim,  gymtorch.unwrap_tensor(self.root_tensor))
        self.step_simulation(50)

    def process_shapenet_objects(self, data_dir="shapenet_training_data"):
        import json

        self.obj_scales = []
        self.obj_paths = []
        self.obj_names = []
        self.obj_ids = []
        for object in os.listdir(data_dir):
            print(object)
            for sample in os.listdir(f'{data_dir}/{object}'):
                filename = f'{data_dir}/{object}/{sample}'
                with open(filename) as json_file:
                    data = json.load(json_file)
                    # print(filename)
                    # print(data.keys())
                    
                    self.obj_ids.append(data['id'])
                    self.obj_names.append(data['category'])
                    self.obj_paths.append(data['path'])
                    self.obj_scales.append(data['scale'])

    def process_grasps(self, file):
        # import pickle
        import pickle5 as pickle
        with open(file, "rb") as fh:
            self.data = pickle.load(fh)
        # file_to_read = open(file, "rb")
        # self.data = pickle.load(file_to_read)

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
            self.scale = [1]*len(self.qualities_1)

        if self.num_envs == 0:
            self.num_envs = len(self.qualities_1)

        # self.isaac_labels = torch.zeros(len(self.quaternions)).to(self.device)


        # if self.quality_type == "top1" or self.quality_type == "bottom1":
        #     self.quality = data['qualities_1']
        # elif self.quality_type == "top2" or self.quality_type == "bottom2":
        #     self.quality = data['qualities_2']
        # else:
        #     print("error in choosing quality")

    def process_grasps_filter(self, file):
        # import pickle
        import pickle5 as pickle
        with open(file, "rb") as fh:
            data = pickle.load(fh)
        # file_to_read = open(file, "rb")
        # data = pickle.load(file_to_read)

        self.qualities_1 = np.array(data['qualities_1'])
        self.qualities_2 = np.array(data['qualities_2'])
        self.isaac_labels = torch.zeros(len(self.qualities_2)).to(self.device)



        self.obj_name = data['fname']
        self.scale = data['scale']
        self.obj_pose_relative = data['obj_pose_relative']
        
        
        self.transforms = data['transforms']
        self.translations = data['translations']
        self.quaternions = data['quaternions']
        self.collisions = data['collisions']


        if self.quality_type == "top1" or self.quality_type == "bottom1":
            self.quality = data['qualities_1']
        elif self.quality_type == "top2" or self.quality_type == "bottom2":
            self.quality = data['qualities_2']
        elif self.quality_type == 'None':
            self.idxs_to_test = np.zeros(self.qualities_1.shape[0])
            self.idxs_to_test = np.where(np.logical_or(self.qualities_1 != 0, self.qualities_2 != 0))[0]


            self.transforms = data['transforms'][self.idxs_to_test]
            self.translations = data['translations'][self.idxs_to_test]
            self.quaternions = data['quaternions'][self.idxs_to_test]
            self.collisions = data['collisions'][self.idxs_to_test]
            self.qualities_1 = data['qualities_1'][self.idxs_to_test]
            self.qualities_2 = data['qualities_2'][self.idxs_to_test]

            print(f"Only {len(self.idxs_to_test)} grasps to test.")
            self.num_envs = len(self.idxs_to_test)
        else:
            print("error in choosing quality")






        # print(f"obj_name: {self.obj_name}")
        # print(f"transforms: {self.transforms.shape}")
        # print(f"translations: {self.translations.shape}")
        # print(f"quaternions: {self.quaternions.shape}")
        # print(f"collisions: {len(self.collisions)}")
        # print(f"qualities_1: {len(self.qualities_1)}")
        # print(f"qualities_2: {len(self.qualities_2)}")
        # print(f"obj_pose_relative: {self.obj_pose_relative}")
        # exit()

    def get_gripper_pose(self, translation=[0,0,0], quaternion=[0,0,0,1], transform=None):

        # gripper_pose = gymapi.Transform()
        # gripper_pose.p = gymapi.Vec3(0, 0, 0)
        # gripper_pose.p.x = -0.01
        # gripper_pose.p.y = -0.01
        # # gripper_pose.p.z = self.offset + 0.125
        # gripper_pose.p.z = self.offset + 0.18
        # gripper_pose.r = gymapi.Quat.from_euler_zyx(np.pi, 0, 0)

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

        # print(translation)
        # print(quaternion)
        # print(gripper_pose.r.w)
        # print(gripper_pose.r.x)
        # print(gripper_pose.r.y)
        # print(gripper_pose.r.z)

        # eulers = R.from_matrix(transform[:3,:3]).as_euler('zxy')
        # eulers = np.flip(eulers)
        # eulers[2] +=  (np.pi + np.pi/2)
        # gripper_pose.r = gymapi.Quat()
        # gripper_pose.r = gymapi.Quat.from_euler_zyx(, eulers[1], eulers[0])
        flipping_rot = gymapi.Quat()
        flipping_rot.w = np.cos(-np.pi/4)
        flipping_rot.z = np.sin(-np.pi/4)

        gripper_pose.r = gripper_pose.r * flipping_rot


        return gripper_pose

    def create_gripper_actor(self, env, asset, pose, name, i, filter = 0):

        franka_dof_props = self.gym.get_asset_dof_properties(asset)
        franka_upper_limits = franka_dof_props["upper"]
        # franka_lower_limits = franka_dof_props["lower"]
        # franka_ranges = franka_upper_limits - franka_lower_limits
        # franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)
        # grippers
        franka_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][:].fill(800.0)
        franka_dof_props["damping"][:].fill(40.0)

        # default dof states and position targets
        franka_num_dofs = self.gym.get_asset_dof_count(asset)
        default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32) # initializing the joints degrees
        # grippers open
        default_dof_pos[-2:] = franka_upper_limits[-2:]
        default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # send to torch
        # default_dof_pos_tensor = to_torch(default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        # franka_link_dict = self.gym.get_asset_rigid_body_dict(self.gripper_asset)
        # franka_hand_index = franka_link_dict["panda_hand"]

        self.gripper_handle = self.gym.create_actor(env, asset, pose, name, i, filter)

        # set dof properties
        self.gym.set_actor_dof_properties(env, self.gripper_handle, franka_dof_props)

        # set initial dof states
        self.gym.set_actor_dof_states(env, self.gripper_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        self.gym.set_actor_dof_position_targets(env, self.gripper_handle, default_dof_pos)

        gripper_idx = self.gym.find_actor_rigid_body_index(env, self.gripper_handle, "panda_hand", gymapi.DOMAIN_SIM)
        self.gripper_idxs.append(gripper_idx)

    def create_obj_actor(self, env, asset, pose, name, i, filter = 0):
            
        self.obj_handle = self.gym.create_actor(env, asset, pose, name, i, filter) # i means the collision group that the actor belongs to. 3 means bitwise filter for elements in the same collisionGroup to mask off collision
        # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        # self.gym.set_rigid_body_color(env, self.obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # get global index of box in rigid body state tensor
        obj_idx = self.gym.get_actor_rigid_body_index(env, self.obj_handle, 0, gymapi.DOMAIN_SIM)
        self.obj_idxs.append(obj_idx)

    def move_gripper_up(self, up = 0.025):
        self.pos_action[:,1] = -up
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def move_gripper_down(self, down = 0.025):
        self.pos_action[:,1] = down
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def stop_gripper(self):
        self.pos_action[:,0] = 0
        self.pos_action[:,1] = 0
        # self.pos_action[:,2] = down
        # self.pos_action[:,3] = down
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
        # self.pos_action[:,1] = 0
        # self.pos_action[:,2] = down
        # self.pos_action[:,3] = down
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))

    def y_rotate(self, angle = np.pi/2):
        self.rotate_gripper(angle/2)
        self.step_simulation(15)
        self.rotate_gripper(-angle)
        self.step_simulation(30)
        self.rotate_gripper(angle)
        self.step_simulation(30)
        self.rotate_gripper(-angle)
        self.step_simulation(30)
        self.rotate_gripper(angle/2)
        self.step_simulation(15)
        self.stop_gripper()
        self.step_simulation(30)

    def gripper_shake(self):
        self.z_translate()
        self.y_rotate()
        self.step_simulation(50)

    def close_gripper_effort(self, effort = -15):
        for _ in range(200):
            self.effort_action[:,2] = effort
            self.effort_action[:,3] = effort
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_action))
            self.step_simulation(1)


    def close_gripper(self):
        # while True:
        #     self.pos_action[:,2] = 0
        #     self.pos_action[:,3] = 0
        #     self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_action))
        #     self.step_simulation(50)

        #     if self.gripper_finger_stable():
        #         break


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
        self.step_simulation(100)


    def create_gripper_asset(self, franka_asset_file = "urdf/panda_hand/panda_hand.urdf"):
        # franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
        # franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper_virtual_joint.urdf"
        franka_asset_file = "urdf/panda_hand/panda_hand.urdf"
        gripper_asset = self.create_obj_asset(self.sim, self.asset_root, franka_asset_file, self.gripper_asset_options())
        return gripper_asset

    def create_obj_asset(self, sim, asset_root, asset_file, asset_options):
        asset = self.gym.load_asset(sim, asset_root, asset_file, asset_options)
        return asset
        
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
            # self.gym.refresh_jacobian_tensors(self.sim)
            # self.gym.refresh_mass_matrix_tensors(self.sim)

            # update viewer
            if self.args.headless == 'Off':

                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                self.gym.sync_frame_time(self.sim)

            

            self.get_root_tensor()

            self.obj_pos = self.rb_states[self.obj_idxs, :3]
            self.obj_rot = self.rb_states[self.obj_idxs, 3:7]
            self.obj_vel = self.rb_states[self.obj_idxs, 7:]
            # print(f"self.obj_vel: {self.obj_vel}")


            self.gripper_pos = self.rb_states[self.gripper_idxs, :3]
            self.gripper_rot = self.rb_states[self.gripper_idxs, 3:7]
            self.gripper_vel = self.rb_states[self.gripper_idxs, 7:]   
            # print(f"self.dof_vel: {self.dof_vel[:,2:,0]}")


            self.dof_pos = self.dof_states[:, 0].view(self.num_envs, -1, 1)
            self.dof_vel = self.dof_states[:, 1].view(self.num_envs, -1, 1)


            # actor Get body state information
            # self.obj_states = self.gym.get_actor_rigid_body_states(self.env, self.obj_handle, gymapi.STATE_ALL)
            # print(f"self.obj_states: {self.obj_states}")

            # # Print some state slices
            # print("Poses from Body State:")
            # print(body_states['pose'])          # print just the poses

            # print("\nVelocities from Body State:")
            # print(body_states['vel'])          # print just the velocities
            # print()




    def obj_asset_options(self):
        obj_asset_options = gymapi.AssetOptions()
        obj_asset_options.disable_gravity = True
        # obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
        obj_asset_options.vhacd_enabled = True
        # obj_asset_options.vhacd_enabled = False
        # obj_asset_options.convex_decomposition_from_submeshes = True
        obj_asset_options.override_com = True
        obj_asset_options.override_inertia = True
        return obj_asset_options

    def gripper_asset_options(self):
        gripper_asset_options = gymapi.AssetOptions()
        gripper_asset_options.armature = 0.1
        gripper_asset_options.fix_base_link = True
        # gripper_asset_options.fix_base_link = False
        gripper_asset_options.disable_gravity = True
        gripper_asset_options.flip_visual_attachments = False
        return gripper_asset_options

    def load_as_urdf(self, obj_name=None, asset_dir=None, obj_path=None, texture_obj_path=None, mass=0.1, scale=1):

        if texture_obj_path is None:
            texture_obj_path = obj_path
        if obj_name is None:
            obj_name = 'unknown'
                
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
        # save_dir = f"{asset_dir}/{obj_path.rsplit('/', 1)[0]}"
        if self.dataset == 'ycb':
            save_file = f"urdf/{obj_path.rsplit('/', 1)[0]}/temp_{obj_name}.urdf"
        elif self.dataset == 'boxes':
            save_file = f"{obj_path.rsplit('/', 1)[0]}/temp_{obj_name}.urdf"
        elif self.dataset == 'shapenet':
            save_file = f"{obj_path.rsplit('/', 1)[0]}/temp_{obj_name}.urdf"
        save_path = f'{asset_dir}/{save_file}'
        urdf_file = open(save_path, 'w')
        urdf_file.write(urdf_txt)

        return save_file





    def get_sim_params(self):
        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
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
