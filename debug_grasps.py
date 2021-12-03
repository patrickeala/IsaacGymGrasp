import IsaacGymSimulator as isaac_sim


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import numpy as np

if __name__ == "__main__":
    # Add custom arguments
    custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of environments to create"},
        {"name": "--object", "type": str, "default": "025_mug", "help": "Object name as in YCB dataset"},
        {"name": "--quality_type", "type": str, "default": "None", "help": "Choose from ['top1', 'top2', 'bottom1', 'bottom2']"},
        {"name": "--headless", "type": str, "default": "Off", "help": "Headless=On has no graphics but faster simulations"},
        {"name": "--dataset", "type": str, "default": "boxes", "help": "'shapenet', 'boxes', 'ycb'"},
    ]
    args = gymutil.parse_arguments(
        description="test",
        custom_parameters=custom_parameters,
    )
    isaac = isaac_sim.IsaacGymSim(args)

    # ===== Creating obj asset =====
    isaac.obj_asset_file = isaac.load_as_urdf(obj_name="mug", asset_dir=isaac.obj_asset_root, obj_path=isaac.obj_path, texture_obj_path=isaac.obj_path, scale=1)
    isaac.obj_asset = isaac.create_obj_asset(isaac.sim, isaac.obj_asset_root, isaac.obj_asset_file, isaac.obj_asset_options())

    # ===== Creating gripper asset =====
    isaac.gripper_asset = isaac.create_gripper_asset()

    isaac.offset = 0

    isaac.create_envs(num_envs=isaac.num_envs)



    position = np.array([ 0.5375109301, 0.084956427533, 0.40107295353 ])
    quaternion = [ -0.00847525323998, 0.999286695659, -0.0272022701815, -0.0247852064345]
    file = '/home/user/isaacgym/python/IsaacGymGrasp/debug/transform_for_tomato_soup_can.pkl'

    import pickle
    with open(file, "rb") as fh:
        reg_transform = pickle.load(fh, encoding='latin1')



    from scipy.spatial.transform import Rotation as R


    # r = R.from_quat(quaternion)
    # rot = r.as_matrix()
    # trans = np.hstack([rot, position.reshape(3,1)])
    # last = np.array([0,0,0,1])
    # trans = np.vstack([trans,last])
    
    rot = np.eye(4)

    rot[:3,:3] = R.from_quat(quaternion).as_matrix()
    rot[:3,3] = position



    rot = np.matmul(reg_transform, rot)
    # rot = np.matmul(rot, reg_transform)

    quat = R.from_matrix(rot[:3,:3]).as_quat()

    path = "/home/user/isaacgym/python/IsaacGymGrasp/isaac_test_10_meshes/004_sugar_box.pkl"
    import pickle5 as pickle
    
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    offset = data['obj_pose_relative']

    # print(reg_transform)
    for i in range(isaac.num_envs):
        env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
        isaac.envs.append(env)

        # ===== Object pose  =====
        isaac.obj_pose = isaac.get_object_pose([0,0,0])
        # isaac.obj_pose = isaac.get_object_pose()
        # isaac.obj_pose.p.x -= offset[0]
        # isaac.obj_pose.p.y -= offset[1]
        # isaac.obj_pose.p.z -= offset[2]
        isaac.create_obj_actor(env, isaac.obj_asset, isaac.obj_pose, "obj", i, 0)
        idx = i%len(isaac.quaternions)
        # idx = 0


        isaac.gripper_pose = gymapi.Transform()
        isaac.gripper_pose.p = gymapi.Vec3(0, 0, 0)
        isaac.gripper_pose.p.x = rot[0, 3] 
        isaac.gripper_pose.p.y = rot[1, 3] 
        isaac.gripper_pose.p.z = rot[2, 3] 

        isaac.gripper_pose.r = gymapi.Quat()
        isaac.gripper_pose.r.x = quat[0]
        isaac.gripper_pose.r.y = quat[1]
        isaac.gripper_pose.r.z = quat[2]
        isaac.gripper_pose.r.w = quat[3]



        flipping_rot = gymapi.Quat()
        flipping_rot.w = np.cos(-np.pi/4)
        flipping_rot.z = np.sin(-np.pi/4)
        isaac.gripper_pose.r = isaac.gripper_pose.r * flipping_rot






        # isaac.gripper_pose = isaac.get_gripper_pose(position, quaternion)
        isaac.create_gripper_actor(env, isaac.gripper_asset, isaac.gripper_pose, "panda_gripper", i, 2)

    isaac.set_camera_pos()


    isaac.prepare_tensors()
    # isaac.step_simulation(2000)
    isaac.step_simulation(50)
    isaac.move_gripper_away()
    
    
    # isaac.move_obj_to_pos()



    isaac.move_gripper_to_grasp()
    isaac.close_gripper()
    isaac.gripper_shake()
    isaac.step_simulation(100)

    isaac.isaac_labels = isaac.check_grasp_success_pos()
    # isaac.save_isaac_labels()        
    isaac.print_results()
    isaac.step_simulation(200)
    isaac.cleanup()