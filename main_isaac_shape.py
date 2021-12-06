import IsaacGymSimulator as isaac_sim


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R

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


    isaac.set_paths()
    # isaac.process_grasps(isaac.grasp_file)
    isaac.process_shapenet_objects()

    # ===== Creating gripper asset =====
    isaac.gripper_asset = isaac.create_gripper_asset()

    isaac.offset = 0

    isaac.create_envs(num_envs=isaac.num_envs)

    for i in range(isaac.num_envs):
        env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
        isaac.envs.append(env)

        # idx = 2
        idx = i % len(isaac.obj_names)

        obj_scale = isaac.obj_scales[idx]
        obj_path = isaac.obj_paths[idx]
        obj_name = isaac.obj_names[idx]
        print(obj_name)


        # ===== Creating obj asset/actor =====
        obj_trans = [0,0,0]
        obj_quat = R.from_euler('zyx', [0, 90, 0], degrees=True).as_quat()
        isaac.obj_pose = isaac.get_object_pose(obj_trans, obj_quat)
        obj_asset_file = isaac.load_as_urdf(obj_name=obj_name, asset_dir=isaac.obj_asset_root,
                                            obj_path=obj_path, texture_obj_path=obj_path,
                                            scale=obj_scale)
        obj_asset = isaac.create_obj_asset(isaac.sim, isaac.obj_asset_root,
                                                obj_asset_file,
                                                isaac.obj_asset_options())
        isaac.create_obj_actor(env, obj_asset, isaac.obj_pose, obj_name, i, 0)

        # ===== Creating gripper actor =====

        grip_trans = [0.02,0,0.15]
        grip_quat = R.from_euler('zyx', [0, 0, 180], degrees=True).as_quat()
        isaac.gripper_pose = isaac.get_gripper_pose(grip_trans, grip_quat)
        isaac.create_gripper_actor(env, isaac.gripper_asset, isaac.gripper_pose, "panda_gripper", i, 2)



    isaac.set_camera_pos()


    isaac.prepare_tensors()
    # isaac.step_simulation(2000)
    isaac.step_simulation(1)



    # ===== executing grasp =====
    # isaac.step_simulation(50)
    # isaac.move_gripper_away()
    # isaac.move_obj_to_pos()
    # isaac.move_gripper_to_grasp()
    isaac.close_gripper()
    isaac.gripper_shake()
    isaac.step_simulation(100)


    isaac.isaac_labels = isaac.check_grasp_success_pos()
    # isaac.save_isaac_labels()        
    isaac.print_results()
    isaac.step_simulation(200)
    isaac.cleanup()