import IsaacGymSimulator as isaac_sim


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

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

    for i in range(isaac.num_envs):
        env = isaac.gym.create_env(isaac.sim, isaac.env_lower, isaac.env_upper, isaac.num_per_row)
        isaac.envs.append(env)

        # ===== Object pose  =====
        isaac.obj_pose = isaac.get_object_pose()
        isaac.create_obj_actor(env, isaac.obj_asset, isaac.obj_pose, "mug", i, 0)
        idx = i%len(isaac.quaternions)
        # idx = 0
        isaac.gripper_pose = isaac.get_gripper_pose(isaac.translations[idx], isaac.quaternions[idx], isaac.transforms[idx, :, :])
        isaac.create_gripper_actor(env, isaac.gripper_asset, isaac.gripper_pose, "panda_gripper", i, 2)

    isaac.set_camera_pos()


    isaac.prepare_tensors()
    isaac.step_simulation(2000)

    isaac.execute_grasp()
    isaac.isaac_labels = isaac.check_grasp_success_pos()
    isaac.save_isaac_labels()        
    isaac.print_results()
    isaac.step_simulation(200)
    isaac.cleanup()